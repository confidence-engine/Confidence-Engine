#!/usr/bin/env python3
import os
from dotenv import load_dotenv

# Load environment variables
print("Loading .env file...")
load_dotenv()

print("\n=== Environment Variables Check ===")

# Check numbered keys
for i in range(1, 6):
    key_name = f"PPLX_API_KEY_{i}"
    key_value = os.getenv(key_name, "").strip()
    if key_value:
        print(f"{key_name}: {key_value[:25]}...")
    else:
        print(f"{key_name}: NOT SET")

# Check single key
single_key = os.getenv("PPLX_API_KEY", "").strip()
if single_key:
    print(f"PPLX_API_KEY: {single_key[:25]}...")
else:
    print("PPLX_API_KEY: NOT SET")

print("\n=== File Contents Check ===")
# Read .env file directly
try:
    with open('.env', 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('PPLX_API_KEY'):
                key, value = line.split('=', 1)
                print(f"File: {key.strip()}: {value.strip()[:25]}...")
except Exception as e:
    print(f"Error reading .env file: {e}")
