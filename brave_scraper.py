import requests
import json
from pprint import pprint
import os
from dotenv import load_dotenv
from lxml import html

# Load environment variables from a .env file.
# This makes the script more secure by not hard-coding sensitive information.
load_dotenv()

# --- Brave Search API Configuration ---
# Get the API key from the environment variable.
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")
BRAVE_API_URL = "https://api.search.brave.com/res/v1/web/search"

def brave_search(query):
    """
    Performs a web search using the Brave Search API.

    Args:
        query (str): The search query string.

    Returns:
        dict: The JSON response from the Brave Search API, or None on failure.
    """
    # Check if the API key was loaded successfully.
    if not BRAVE_API_KEY:
        print("Error: Brave API key not found. Please set it in a .env file.")
        return None

    # The headers for the API request, including the required 'X-Subscription-Token'
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": BRAVE_API_KEY,
    }

    # The parameters for the search query
    params = {
        "q": query,
    }

    try:
        print(f"Searching for: '{query}'...")
        # Make the GET request to the Brave Search API
        response = requests.get(BRAVE_API_URL, headers=headers, params=params)

        # Raise an exception for bad status codes (4xx or 5xx)
        response.raise_for_status()

        # Parse the JSON response
        results = response.json()
        print("Search successful!")
        return results

    except requests.exceptions.HTTPError as errh:
        print(f"HTTP Error: {errh}")
    except requests.exceptions.ConnectionError as errc:
        print(f"Connection Error: {errc}")
    except requests.exceptions.Timeout as errt:
        print(f"Timeout Error: {errt}")
    except requests.exceptions.RequestException as err:
        print(f"An unexpected error occurred: {err}")
    
    return None

def scrape_page_content(url):
    """
    Scrapes the content of a given URL using requests and lxml.

    Args:
        url (str): The URL of the web page to scrape.

    Returns:
        str: The extracted text content from the page, or a message on failure.
    """
    try:
        print(f"\nScraping content from: {url}")
        # Use requests to get the HTML content of the page
        page_response = requests.get(url, timeout=10)
        page_response.raise_for_status()

        # Parse the HTML content using lxml
        # page_response.content contains the raw bytes of the response
        tree = html.fromstring(page_response.content)

        # Use XPath to find all paragraph (<p>) tags and get their text
        # This is a simple example. You can create much more complex XPath
        # selectors to target specific elements on a page.
        paragraphs = tree.xpath('//p/text()')

        # Join the paragraphs into a single string
        content = "\n".join(paragraphs).strip()
        
        if content:
            print("Successfully extracted content.")
            return content
        else:
            return "No paragraph content found."

    except requests.exceptions.RequestException as e:
        return f"Error fetching page: {e}"
    except Exception as e:
        return f"An error occurred during scraping: {e}"


def main():
    """
    Main function to run the Brave Search and scrape the first result.
    """
    search_query = "latest news in artificial intelligence"
    search_results = brave_search(search_query)

    if search_results:
        # The API response is a dictionary. The search results are under the 'web' key.
        web_results = search_results.get("web", {}).get("results", [])

        if not web_results:
            print("\nNo web results found for the query.")
            return

        print("\n--- Displaying Top Web Results ---")
        for i, result in enumerate(web_results):
            # Extract key information from each result
            title = result.get("title", "No Title")
            url = result.get("url", "No URL")
            snippet = result.get("description", "No snippet available")

            print(f"\nResult {i+1}:")
            print(f"  Title: {title}")
            print(f"  URL: {url}")
            print(f"  Snippet: {snippet[:200]}...") # Truncate for cleaner output

            # --- LXML Scraping Integration ---
            # For demonstration, let's scrape the content of the first result
            if i == 0 and url != "No URL":
                scraped_content = scrape_page_content(url)
                print("\n--- Scraped Content from First Result ---")
                # Print the first 500 characters of the scraped content
                print(scraped_content[:500] + "..." if len(scraped_content) > 500 else scraped_content)
        
        # You can also pretty-print the entire JSON response for a full view
        # print("\n--- Full JSON Response (for debugging) ---")
        # pprint(search_results)

if __name__ == "__main__":
    # Before running this script, you must install the required libraries:
    # pip install requests python-dotenv lxml
    main()
