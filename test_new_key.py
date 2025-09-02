#!/usr/bin/env python3
import os
import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_single_key(api_key, key_name):
    """Test a single Perplexity API key"""
    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "sonar",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello"}
        ],
        "temperature": 0.1
    }

    print(f"Testing {key_name}: {api_key[:25]}...")
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.post(url, headers=headers, json=payload)
            print(f"Status: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
                print("✅ SUCCESS!")
                print(f"Response: {content[:100]}...")
                return True
            elif response.status_code == 401:
                print("❌ FAILED: 401 Unauthorized")
                print(f"Response: {response.text[:200]}")
                return False
            elif response.status_code == 429:
                print("❌ FAILED: 429 Rate Limited")
                return False
            else:
                print(f"❌ FAILED: {response.status_code}")
                print(f"Response: {response.text[:200]}")
                return False
    except Exception as e:
        print(f"❌ EXCEPTION: {str(e)}")
        return False

def main():
    print("🔍 Testing Updated PPLX_API_KEY_1\n")

    # Test the updated key
    key = os.getenv("PPLX_API_KEY_1", "").strip()
    if key:
        success = test_single_key(key, "PPLX_API_KEY_1")
        if success:
            print("\n🎉 NEW KEY WORKS! You can now update the other keys or re-enable Perplexity sentiment.")
        else:
            print("\n❌ New key still doesn't work. Check if it's correct or try generating another one.")
    else:
        print("❌ PPLX_API_KEY_1 is empty")

if __name__ == "__main__":
    main()
