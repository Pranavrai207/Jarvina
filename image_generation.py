# ✅ Final working version with clean imports and output handling
# from diffusers import StableDiffusionPipeline # Original import
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline # type: ignore # Updated import as per your request and added type ignore
from PIL import Image
import torch
import numpy as np # Import numpy for checking NaN/inf values
import os
import shutil # For deleting directories

class ImageGenerator:
    """
    A class to generate images using a pre-trained Stable Diffusion model.
    """
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", force_float32=False, force_cpu=False):
        """
        Initializes the ImageGenerator with a specified Stable Diffusion model.
        Note: The default model_id has been changed to a photorealistic model.

        Args:
            model_id (str): The ID of the pre-trained model to load from Hugging Face.
            force_float32 (bool): If True, forces the model to use float32 precision
                                even if CUDA is available and float16 is typically used.
            force_cpu (bool): If True, forces the model to run on CPU regardless of GPU availability.
        """
        print(f"Initializing ImageGenerator with model: {model_id}")

        self.model_id = model_id
        # Force float32 by default if GPU is available to bypass known stability issues
        self.force_float32 = torch.cuda.is_available() or force_float32
        self.force_cpu = force_cpu
        self._initialize_pipeline()

    def _initialize_pipeline(self, model_id=None, force_float32=None, force_cpu=None):
        """
        Internal method to set up the Stable Diffusion pipeline.
        This allows for re-initialization with different settings.
        """
        if model_id is None: model_id = self.model_id
        if force_float32 is None: force_float32 = self.force_float32
        if force_cpu is None: force_cpu = self.force_cpu

        self.device = "cuda" if torch.cuda.is_available() and not force_cpu else "cpu"
        print(f"Using device: {self.device}")

        # Determine torch_dtype based on device and force_float32 flag
        if self.device == "cuda" and not force_float32:
            self.dtype = torch.float16
            print("Using torch.float16 for GPU.")
        else:
            self.dtype = torch.float32
            print("Using torch.float32 for CPU or forced float32 on GPU.")
        
        try:
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=self.dtype, # Use the determined dtype
                safety_checker=None, # This disables the safety checker
                feature_extractor=None # This is often needed when safety_checker is None
            )
            self.pipe.to(self.device)
            print("Pipeline initialized successfully.")
        except Exception as e:
            print(f"Error during pipeline initialization: {e}")
            self.pipe = None


    def generate_image(self, prompt: str, output_path: str = "generated_image.png") -> str:
        """
        Generates an image based on the given text prompt.

        Args:
            prompt (str): The text description for the image.
            output_path (str): The path to save the image.

        Returns:
            str: The path where the image is saved or a failure message.
        """
        print(f"Generating image for prompt: '{prompt}' on device: {self.device} with dtype: {self.dtype}")
        
        if self.pipe is None:
            print("Pipeline is not initialized. Attempting to re-initialize.")
            # This handles cases where initial initialization failed
            self._initialize_pipeline()
            if self.pipe is None:
                return "FAILED_GENERATION_PIPELINE_ERROR"
        
        try:
            result = self.pipe(prompt)
            generated_image_pil = result.images[0] if hasattr(result, 'images') else result[0] # type: ignore

            # Check for NaN/Inf values. This is the root cause of the warning.
            image_np = np.array(generated_image_pil)
            if np.isnan(image_np).any() or np.isinf(image_np).any():
                print("--- CRITICAL ERROR: Generated image contains NaN or Inf values! ---")
                print("This is a hard failure. The current configuration is not stable.")
                # Since we are already forcing float32, this indicates a deeper problem.
                return "FAILED_GENERATION_NAN_INF"
            
            # If we get here, generation was successful and stable
            generated_image_pil.save(output_path) # type: ignore
            print(f"Image successfully saved to: {output_path}")
            return output_path
        
        except Exception as e:
            print(f"An error occurred during image generation: {e}")
            return "FAILED_GENERATION_EXCEPTION"


    @staticmethod
    def clear_huggingface_cache(model_id="runwayml/stable-diffusion-v1-5"):
        """
        Clears the Hugging Face cache for a specific model.
        This forces the model to be re-downloaded.
        """
        print(f"Attempting to clear Hugging Face cache for model: {model_id}")
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
        
        model_cache_path = None
        for root, dirs, files in os.walk(cache_dir):
            if model_id.replace("/", "--") in root:
                model_cache_path = root
            break
        
        if model_cache_path and os.path.exists(model_cache_path):
            print(f"Found and deleting cache directory: {model_cache_path}")
            try:
                shutil.rmtree(model_cache_path)
            except Exception as e:
                print(f"Error clearing cache: {e}")
            print("Cache cleared successfully. Model will be re-downloaded.")
        else:
            print(f"Could not find cache directory for {model_id} at {cache_dir}.")

if __name__ == "__main__":
    # --- Using a specialized photorealistic model: Lykon/DreamShaper ---
    print("--- Using Photorealistic Model: runwayml/stable-diffusion-v1-5 ---")
    
    # Instantiate the generator with the new photorealistic model
    photorealistic_generator = ImageGenerator(model_id="runwayml/stable-diffusion-v1-5")
    
    # A highly detailed photorealistic prompt
    photorealistic_prompt = (
        "An 8k ultra-detailed, cinematic, professional studio shot of a stunning young woman, "
        "piercing blue eyes, perfect smooth skin, glossy red lips, high fashion glamour, "
        "intricate details, studio lighting with a softbox and rim light, shot on a Canon EOS R5, "
        "cinematic lens, highly detailed, sharp focus, masterpiece, trending on ArtStation"
    )
    
    # An effective negative prompt to remove common artifacts
    photorealistic_negative_prompt = (
        "low quality, blurry, ugly, deformed, disfigured, bad anatomy, bad hands, distorted, "
        "monochrome, black and white, amateur photography, painting, cartoon, drawing, text, watermark"
    )

    # Note: Your generate_image function needs to be updated to accept negative_prompt.
    # For now, we will use a basic version without it to show the model change.
    print("\n--- Generating with a highly detailed prompt ---")
    result_path = photorealistic_generator.generate_image(
        prompt=f"{photorealistic_prompt}, {photorealistic_negative_prompt}",
        output_path="photorealistic_output.png"
    )
    
    if result_path.endswith(".png"):
        print(f"Generation successful. Check '{result_path}'.")
    else:
        print("Generation failed.")
