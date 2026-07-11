import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if api_key:
    print(f"Key loaded successfully! Starts with: {api_key[:10]}...")
else:
    print("Key not found. Check your .env file.")