import os
import json
import re
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ConfigDict
from openrouter_client import OpenRouterClient
from dotenv import load_dotenv
from typing import Optional, Union
from datetime import datetime
import logging
import httpx

# --- Import your custom ImageGenerator class ---
from image_generator import ImageGenerator

# Load environment variables from .env file
load_dotenv()

# --- Global Variables & Constants ---
openrouter_client: Optional[OpenRouterClient] = None
jarvina_persona_data: Optional[dict] = None
custom_replies_map: dict = {}
image_generator: Optional[ImageGenerator] = None # Global variable for the image generator instance
CUSTOM_INSTRUCTIONS_FILE = "custom_instructions.json"
# --- UPDATED: Changed the directory to 'static' as requested ---
GENERATED_IMAGES_DIR = "static"

# --- Create the directory for generated images BEFORE the app is initialized ---
os.makedirs(GENERATED_IMAGES_DIR, exist_ok=True)

# Define models for different use cases
FAIL_SAFE_MODEL = "mistralai/mistral-7b-instruct-v0.2"
EMOTIONAL_MODEL = "openai/gpt-4o"
ANALYSIS_MODEL = "anthropic/claude-3-sonnet"
CODING_MODEL = "google/gemini-pro"
GENERAL_CHAT_MODEL = "mistralai/mistral-7b-instruct-v0.2"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Helper function to normalize text
def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Functions for custom instructions
def load_custom_instructions_from_file() -> str:
    if os.path.exists(CUSTOM_INSTRUCTIONS_FILE):
        try:
            with open(CUSTOM_INSTRUCTIONS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f).get("instructions", "")
        except Exception as e:
            logging.error(f"Error loading {CUSTOM_INSTRUCTIONS_FILE}: {e}")
    return ""

def save_custom_instructions_to_file(instructions: str):
    try:
        with open(CUSTOM_INSTRUCTIONS_FILE, 'w', encoding='utf-8') as f:
            json.dump({"instructions": instructions}, f, indent=4)
    except Exception as e:
        logging.error(f"Error saving to {CUSTOM_INSTRUCTIONS_FILE}: {e}")
        raise

# --- Lifespan Event Handler ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global openrouter_client, jarvina_persona_data, custom_replies_map, image_generator

    # Initialize OpenRouter Client
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set.")
    openrouter_client = OpenRouterClient(api_key=api_key)
    print("OpenRouterClient initialized successfully.")

    # Load persona data and custom replies
    memory_file_path = "memory.json"
    if os.path.exists(memory_file_path):
        try:
            with open(memory_file_path, 'r', encoding='utf-8') as f:
                memory_data = json.load(f)
            jarvina_persona_data = memory_data.get("persona_data")
            if jarvina_persona_data:
                print("Persona data loaded successfully.")
            loaded_custom_replies = memory_data.get("custom_replies", [])
            for reply_entry in loaded_custom_replies:
                response = reply_entry.get("response")
                phrases = reply_entry.get("phrases", [])
                if response and phrases:
                    for phrase in phrases:
                        custom_replies_map[normalize_text(phrase)] = response
            if custom_replies_map:
                print("Custom replies loaded successfully.")
        except Exception as e:
            print(f"Error loading {memory_file_path}: {e}")
            jarvina_persona_data = None
            custom_replies_map = {}
    else:
        print(f"Warning: {memory_file_path} not found.")
        jarvina_persona_data = None
        custom_replies_map = {}

    # --- Instantiate the ImageGenerator on startup ---
    print("Initializing ImageGenerator (Stable Diffusion)...")
    try:
        image_generator = ImageGenerator(model_id="runwayml/stable-diffusion-v1-5")
        if image_generator.pipe is None:
            print("CRITICAL ERROR: ImageGenerator initialized, but the pipeline inside it failed to load. Image generation will be disabled.")
            image_generator = None
        else:
            print("ImageGenerator (Stable Diffusion) initialized successfully.")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to instantiate ImageGenerator class: {e}")
        image_generator = None

    yield
    print("FastAPI application shutting down.")

# Initialize FastAPI app and pass the lifespan manager
app = FastAPI(lifespan=lifespan)

# --- UPDATED: Mount the 'static' directory ---
app.mount("/static", StaticFiles(directory=GENERATED_IMAGES_DIR), name="static")

templates = Jinja2Templates(directory="templates")

# Pydantic models
class ChatRequest(BaseModel):
    messages: Optional[list[dict]] = None
    prompt: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    model_config = ConfigDict(extra='allow')

class CustomInstructionsRequest(BaseModel):
    instructions: str

# --- GENERAL CHAT ENDPOINT ---
@app.post("/api/chat/mistral/")
async def chat_endpoint(chat_request: ChatRequest):
    if openrouter_client is None:
        raise HTTPException(status_code=500, detail="OpenRouterClient not initialized.")

    logging.info(f"Received payload: {chat_request.model_dump_json(indent=2)}")

    messages_to_send = []
    user_prompt_content = ""
    frontend_system_message = None
    system_content_parts = []

    if chat_request.messages and chat_request.messages:
        messages_to_send = chat_request.messages
        user_prompt_content = next((msg["content"] for msg in reversed(messages_to_send) if msg.get("role") == "user" and msg.get("content")), "")
        frontend_system_message = next((msg for msg in messages_to_send if msg.get("role") == "system"), None)
    elif chat_request.prompt and chat_request.prompt.strip():
        user_prompt_content = chat_request.prompt
        messages_to_send = [{"role": "user", "content": user_prompt_content}]
    else:
        raise HTTPException(status_code=400, detail="No valid 'messages' or 'prompt' found.")

    normalized_user_input = normalize_text(user_prompt_content)
    logging.info(f"Normalized user input: '{normalized_user_input}'")

    image_gen_keywords = ["generate image", "create a picture", "make a photo", "draw a picture of", "generate a photo of"]
    if any(keyword in normalized_user_input for keyword in image_gen_keywords):
        if image_generator is None:
            return JSONResponse(content={"response": "I'm sorry, the image generation service is currently unavailable."})

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"generated_image_{timestamp}.png" # Added prefix for clarity
        output_filepath = os.path.join(GENERATED_IMAGES_DIR, filename)
        
        clean_prompt = re.sub(r'(generate|create|make an?)\s(image|picture|photo)\s(of|for)?', '', user_prompt_content, flags=re.IGNORECASE).strip()

        logging.info(f"Calling ImageGenerator for prompt: '{clean_prompt}'")
        result_path = image_generator.generate_image(prompt=clean_prompt, output_path=output_filepath)

        if result_path and result_path.endswith(".png"):
            # --- UPDATED: Use the new '/static/' path ---
            web_accessible_path = f"/static/{filename}"
            return JSONResponse(content={
                "response": "Of course. Here is the image you requested.",
                "image_url": web_accessible_path
            })
        else:
            logging.error(f"Image generation failed with code: {result_path}")
            return JSONResponse(content={"response": "I'm sorry, I encountered an error while trying to create the image."})
    
    if normalized_user_input in custom_replies_map:
        return JSONResponse(content={"response": custom_replies_map[normalized_user_input]})

    if "current time" in normalized_user_input or "what time is it" in normalized_user_input:
        current_time_str = datetime.now().strftime('%I:%M:%S %p on %A, %B %d, %Y')
        return JSONResponse(content={"response": f"The current time is {current_time_str}."})

    heard_only_keywords = ["i feel", "just needed", "don’t know why", "venting", "no need to reply", "just saying", "i just want to talk", "i need to get this off my chest"]
    reply_expected_keywords = ["what do you think", "can you tell", "why", "how", "should i", "what if", "explain", "describe", "what is", "tell me about", "elaborate", "discuss", "analyze", "provide details", "in depth", "give me information", "can you tell me", "can you explain", "define", "compare", "contrast", "implications", "effects", "impact", "significance", "describe your thoughts on", "what are your views on", "meaning of", "definition of"]
    emotional_keywords_general = ["feeling", "feel", "sad", "happy", "anxious", "stressed", "emotional", "how are you feeling", "depressed", "overwhelmed", "frustrated", "lonely", "joyful", "excited", "upset", "down", "confused", "worried", "scared", "angry", "hopeful", "grateful", "content", "my mood is"]
    analysis_keywords = ["analyze", "data", "report", "statistics", "trend", "chart", "graph", "metrics", "predict", "forecast", "correlation", "distribution", "summary", "breakdown"]
    coding_keywords = ["code", "program", "script", "function", "bug", "error", "syntax", "develop", "implement", "debug", "algorithm", "language", "python", "javascript", "html", "css", "java", "c++", "react", "api", "framework", "library", "class", "method", "variable"]
    latest_info_keywords = ["latest", "recent", "news", "current events", "update on", "what's new with", "happening now", "breaking news", "today", "this week", "this month", "this year"]
    
    is_heard_only = any(keyword in normalized_user_input for keyword in heard_only_keywords)
    is_reply_expected = any(keyword in normalized_user_input for keyword in reply_expected_keywords) or user_prompt_content.endswith("?")
    is_emotional_general = any(keyword in normalized_user_input for keyword in emotional_keywords_general)
    is_analysis_query = any(keyword in normalized_user_input for keyword in analysis_keywords)
    is_coding_query = any(keyword in normalized_user_input for keyword in coding_keywords)
    is_latest_info_query = any(keyword in normalized_user_input for keyword in latest_info_keywords)
    is_explicit_search_command = normalized_user_input.startswith(("search for ", "what is ", "find out about "))
    is_search_query = is_explicit_search_command or is_latest_info_query
    
    model_to_use = GENERAL_CHAT_MODEL
    temperature_to_use = 0.7
    max_tokens_to_use = 2600

    if is_search_query:
        pass
    elif is_heard_only:
        model_to_use = EMOTIONAL_MODEL
        temperature_to_use = 0.9

    final_system_content = "\n".join(system_content_parts)
    if frontend_system_message:
        frontend_system_message["content"] += "\n\n" + final_system_content
        final_messages_for_llm = [frontend_system_message] + [msg for msg in messages_to_send if msg.get("role") != "system"]
    else:
        default_system_message = {"role": "system", "content": final_system_content}
        final_messages_for_llm = [default_system_message] + messages_to_send

    structured_output_instruction = (
        "CRITICAL AND NON-NEGOTIABLE INSTRUCTION: All responses MUST be structured using ONLY plain text..."
    )
    if final_messages_for_llm:
        final_messages_for_llm[0]["content"] = structured_output_instruction + "\n\n" + final_messages_for_llm[0].get("content", "")


    try:
        response_content = openrouter_client.generate_text(
            model=model_to_use,
            messages=final_messages_for_llm,
            temperature=temperature_to_use,
            max_tokens=max_tokens_to_use,
            fallback_model=FAIL_SAFE_MODEL
        )
        return JSONResponse(content={"response": response_content})
    except Exception as e:
        logging.error(f"Error during AI generation: {e}")
        raise HTTPException(status_code=500, detail=f"AI generation failed: {str(e)}")


# --- NEW DEDICATED ENDPOINT FOR IMAGE GENERATION FROM THE FORM ---
@app.post("/generate_image", response_class=HTMLResponse)
async def generate_image_endpoint(request: Request, prompt: str = Form(...)):
    if image_generator is None:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "image_gen_error": "Sorry, the image generation service is currently unavailable."
        })

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"generated_image_{timestamp}.png" # Added prefix for clarity
    output_filepath = os.path.join(GENERATED_IMAGES_DIR, filename)

    logging.info(f"Calling ImageGenerator from dedicated endpoint for prompt: '{prompt}'")
    result_path = image_generator.generate_image(prompt=prompt, output_path=output_filepath)

    if result_path and result_path.endswith(".png"):
        # --- UPDATED: Use the new '/static/' path ---
        web_accessible_path = f"/static/{filename}"
        return templates.TemplateResponse("index.html", {
            "request": request,
            "generated_image_url": web_accessible_path,
            "image_gen_prompt": prompt
        })
    else:
        logging.error(f"Image generation failed with code: {result_path}")
        return templates.TemplateResponse("index.html", {
            "request": request,
            "image_gen_error": "Sorry, I encountered an error while trying to create the image."
        })


# --- Other Endpoints (Custom Instructions, HTML forms) ---
@app.post("/api/save_custom_instructions")
async def save_custom_instructions(request: CustomInstructionsRequest):
    try:
        save_custom_instructions_to_file(request.instructions)
        return JSONResponse(content={"message": "Custom instructions saved successfully!"})
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to save custom instructions.")

@app.get("/api/load_custom_instructions")
async def load_custom_instructions_api():
    instructions = load_custom_instructions_from_file()
    return JSONResponse(content={"instructions": instructions})

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/emotional_talk", response_class=HTMLResponse)
async def emotional_talk_endpoint(request: Request, prompt: str = Form(...)):
    if openrouter_client is None:
        raise HTTPException(status_code=500, detail="OpenRouterClient not initialized.")
    emotional_prompt_content = (
        f"I'm sorry to hear that you're feeling {prompt.lower().replace('i am feeling ', '').replace('i feel ', '')}. "
        f"Please provide several distinct suggestions as a SINGLE, CONTINUOUS numbered list (1., 2., 3., etc.)."
    )
    response_text = openrouter_client.generate_text(
        model=EMOTIONAL_MODEL,
        messages=[{"role": "user", "content": emotional_prompt_content}],
        temperature=0.8,
        max_tokens=2600,
        fallback_model=FAIL_SAFE_MODEL
    )
    return templates.TemplateResponse("index.html", {
        "request": request,
        "emotional_response": response_text,
        "emotional_prompt": prompt
    })

@app.post("/data_analysis", response_class=HTMLResponse)
async def data_analysis_endpoint(request: Request, prompt: str = Form(...)):
    if openrouter_client is None:
        raise HTTPException(status_code=500, detail="OpenRouterClient not initialized.")
    response_text = openrouter_client.generate_text(
        model=ANALYSIS_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=2600,
        fallback_model=FAIL_SAFE_MODEL
    )
    return templates.TemplateResponse("index.html", {
        "request": request,
        "analysis_response": response_text,
        "analysis_prompt": prompt
    })

@app.post("/coding_help", response_class=HTMLResponse)
async def coding_help_endpoint(request: Request, prompt: str = Form(...)):
    if openrouter_client is None:
        raise HTTPException(status_code=500, detail="OpenRouterClient not initialized.")
    response_text = openrouter_client.generate_text(
        model=CODING_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=2600,
        fallback_model=FAIL_SAFE_MODEL
    )
    return templates.TemplateResponse("index.html", {
        "request": request,
        "coding_response": response_text,
        "coding_prompt": prompt
    })
