import os

from dotenv import load_dotenv
load_dotenv()

raw = os.getenv("PPLX_API_KEYS", "")
print("raw_len:", len(raw))
print("raw_repr:", repr(raw))
print("raw_hex:", raw.encode("utf-8").hex())

parts = raw.split(",")
print("parts_len:", len(parts))
print("parts_repr:", [repr(p) for p in parts])
