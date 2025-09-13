#!/usr/bin/env python3
import os
import sys

def test_openai_initialization():
    print("Testing OpenAI initialization...")
    
    # Test 1: Check if openai is importable
    try:
        import openai
        print(f"✅ OpenAI import successful. Version: {openai.__version__}")
    except ImportError as e:
        print(f"❌ OpenAI import failed: {e}")
        return

    # Test 2: Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    print(f"API key present: {'Yes' if api_key else 'No'}")
    print(f"API key length: {len(api_key) if api_key else 0}")
    
    if not api_key:
        print("❌ No API key found")
        return
    
    # Test 3: Try to initialize client with minimal parameters
    try:
        client = openai.OpenAI(api_key=api_key)
        print("✅ OpenAI client initialized successfully")
        return client
    except Exception as e:
        print(f"❌ OpenAI client initialization failed: {e}")
        print(f"Error type: {type(e).__name__}")
        
        # Test 4: Try different initialization methods
        print("\n🔄 Trying alternative initialization...")
        try:
            client = openai.OpenAI()  # Let it use env var automatically
            print("✅ OpenAI client initialized with env var")
            return client
        except Exception as e2:
            print(f"❌ Alternative initialization also failed: {e2}")
            
        return None

if __name__ == "__main__":
    test_openai_initialization()
