#!/usr/bin/env python3
import os
import httpx
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_key_comprehensive(api_key, key_number):
    """Comprehensive test of a Perplexity API key"""
    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Test 1: Simple request
    print(f"\n--- Testing {key_number} ---")
    print(f"Key: {api_key[:25]}...")

    payload = {
        "model": "sonar",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello"}
        ],
        "temperature": 0.1
    }

    try:
        print("Test 1: Basic request...")
        with httpx.Client(timeout=10.0) as client:
            response = client.post(url, headers=headers, json=payload)
            print(f"  Status: {response.status_code}")
            print(f"  Headers: {dict(response.headers)}")

            if response.status_code == 200:
                data = response.json()
                content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
                print(f"  ✅ SUCCESS: {content[:100]}...")
                return True
            elif response.status_code == 401:
                print(f"  ❌ 401 Unauthorized: {response.text[:200]}")
                return False
            elif response.status_code == 429:
                print(f"  ❌ 429 Rate Limited: {response.text[:200]}")
                return False
            else:
                print(f"  ❌ Other error: {response.text[:200]}")
                return False

    except Exception as e:
        print(f"  ❌ Exception: {str(e)}")
        return False

def test_key_different_models(api_key, key_number):
    """Test with different models"""
    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    models_to_test = ["sonar", "sonar-pro", "sonar-reasoning"]

    for model in models_to_test:
        print(f"Test 2: Model '{model}'...")
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "temperature": 0.1
        }

        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.post(url, headers=headers, json=payload)
                print(f"  Model '{model}': {response.status_code}")
                if response.status_code == 200:
                    return True
        except Exception as e:
            print(f"  Model '{model}': Exception - {str(e)}")

    return False

def test_key_minimal_payload(api_key, key_number):
    """Test with minimal payload"""
    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    print("Test 3: Minimal payload...")
    payload = {
        "model": "sonar",
        "messages": [{"role": "user", "content": "Hi"}]
    }

    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.post(url, headers=headers, json=payload)
            print(f"  Minimal payload: {response.status_code}")
            if response.status_code == 200:
                return True
    except Exception as e:
        print(f"  Minimal payload: Exception - {str(e)}")

    return False

def test_key_get_request(api_key, key_number):
    """Test with GET request to check API availability"""
    url = "https://api.perplexity.ai/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
    }

    print("Test 4: GET request to /models...")
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(url, headers=headers)
            print(f"  GET /models: {response.status_code}")
            if response.status_code == 200:
                print(f"  Available models: {response.json()}")
                return True
    except Exception as e:
        print(f"  GET /models: Exception - {str(e)}")

    return False

def main():
    print("🔍 Comprehensive Perplexity API Key Testing\n")

    # Test numbered keys
    for i in range(1, 6):
        key = os.getenv(f"PPLX_API_KEY_{i}", "").strip()
        if key:
            success = test_key_comprehensive(key, f"PPLX_API_KEY_{i}")
            if not success:
                test_key_different_models(key, f"PPLX_API_KEY_{i}")
                test_key_minimal_payload(key, f"PPLX_API_KEY_{i}")
                test_key_get_request(key, f"PPLX_API_KEY_{i}")
            print(f"\n{'='*50}")
        else:
            print(f"PPLX_API_KEY_{i}: NOT SET\n")

    # Test single key
    single_key = os.getenv("PPLX_API_KEY", "").strip()
    if single_key:
        print("Testing PPLX_API_KEY (single):")
        success = test_key_comprehensive(single_key, "PPLX_API_KEY")
        if not success:
            test_key_different_models(single_key, "PPLX_API_KEY")
            test_key_minimal_payload(single_key, "PPLX_API_KEY")
            test_key_get_request(single_key, "PPLX_API_KEY")

if __name__ == "__main__":
    main()
