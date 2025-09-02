#!/usr/bin/env python3
import os
import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_single_key(api_key, key_number):
    """Test a single Perplexity API key"""
    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "sonar-pro",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello"}
        ],
        "temperature": 0.1
    }

    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.post(url, headers=headers, json=payload)
            print(f"Key {key_number}: HTTP {response.status_code}")
            if response.status_code == 200:
                print(f"  ✅ SUCCESS: {response.json().get('choices', [{}])[0].get('message', {}).get('content', '')[:100]}...")
                return True
            else:
                print(f"  ❌ FAILED: {response.text[:200]}")
                return False
    except Exception as e:
        print(f"Key {key_number}: EXCEPTION: {str(e)}")
        return False

def main():
    print("Testing Perplexity API Keys...\n")

    # Test numbered keys
    for i in range(1, 6):
        key = os.getenv(f"PPLX_API_KEY_{i}", "").strip()
        if key:
            print(f"Testing PPLX_API_KEY_{i}: {key[:20]}...")
            test_single_key(key, f"PPLX_API_KEY_{i}")
            print()
        else:
            print(f"PPLX_API_KEY_{i}: NOT SET")
            print()

    # Test single key
    single_key = os.getenv("PPLX_API_KEY", "").strip()
    if single_key:
        print(f"Testing PPLX_API_KEY: {single_key[:20]}...")
        test_single_key(single_key, "PPLX_API_KEY")
        print()

if __name__ == "__main__":
    main()
