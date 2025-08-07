import os
import requests
import pandas as pd
from dotenv import load_dotenv

def fetch_perplexity_analysis(question):
    """
    Queries the Perplexity API with a specific question using a pool of keys,
    rotating to the next key if one fails or is out of credits.
    """
    load_dotenv()
    print(f"--- Fetching Perplexity Analysis for: '{question}' ---")
    
    # Step 1: Load all available keys from the .env file into a list
    api_keys = []
    i = 1
    while True:
        key = os.getenv(f"PERPLEXITY_API_KEY_{i}")
        if key:
            api_keys.append(key)
            i += 1
        else:
            break

    if not api_keys:
        print("  -> Error: No Perplexity API keys found in .env file.")
        return None

    print(f"  -> Found {len(api_keys)} Perplexity API key(s) to use.")

    # Step 2: Define the request payload
    url = "https://api.perplexity.ai/chat/completions"
    payload = {
        "model": "sonar",
        "messages": [{"role": "system", "content": "You are a concise sports analyst."}, {"role": "user", "content": question}]
    }
    
    # Step 3: Loop through the keys until one succeeds
    for key in api_keys:
        print(f"  -> Attempting query with key ending in '...{key[-4:]}'")
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            # Check for client-side errors that indicate a bad key or no credits
            if 400 <= response.status_code < 500:
                print(f"  -> Key failed (Status {response.status_code}). Trying next key...")
                continue # Move to the next key in the list
            
            response.raise_for_status() # Handle other errors (like 5xx server issues)
            
            data = response.json()
            assistant_message = data['choices'][0]['message']['content']
            print("  -> Successfully received response!")
            # Return the analysis as a DataFrame for consistency
            return pd.DataFrame([{'query': question, 'analysis': assistant_message}])
            
        except requests.exceptions.RequestException as e:
            print(f"  -> An error occurred with this key: {e}. Trying next key...")
            continue
    
    print("  -> All API keys failed. Could not retrieve data from Perplexity.")
    return None