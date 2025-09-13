#!/usr/bin/env python3
"""
Test script to check the form submission issue
"""

test_code = '''def fibonacci(n):
    """Calculate fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def main():
    for i in range(10):
        print(f"F({i}) = {fibonacci(i)}")

if __name__ == "__main__":
    main()'''

import requests

# Test the evaluation endpoint
try:
    response = requests.post(
        "http://localhost:3000/evaluate",
        data={
            "problem_description": "Calculate Fibonacci numbers",
            "code": test_code
        }
    )
    
    print("Status Code:", response.status_code)
    print("Response:", response.json())
    
except Exception as e:
    print("Error:", e)
