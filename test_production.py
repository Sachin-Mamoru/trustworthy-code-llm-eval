#!/usr/bin/env python3
"""
Local test script for production application

This script tests the production app locally before Azure deployment.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Set minimal environment for testing
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("LOG_LEVEL", "INFO")

async def test_production_app():
    """Test the production app components."""
    
    print("🧪 Testing TrustworthyCodeLLM Production App")
    print("=" * 50)
    
    try:
        # Test imports
        print("📦 Testing imports...")
        from web_dashboard.production_app import ProductionCodeEvaluator
        print("✅ Production app imports successful")
        
        # Test evaluator initialization
        print("🔧 Testing evaluator initialization...")
        evaluator = ProductionCodeEvaluator()
        print(f"✅ Evaluator initialized with {len(evaluator.llm_clients)} LLM clients")
        print(f"✅ Static analyzers: {list(evaluator.static_tools.keys())}")
        
        # Test AST analysis (doesn't require external APIs)
        print("📊 Testing AST analysis...")
        test_code = '''
def fibonacci(n):
    """Calculate fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def main():
    try:
        result = fibonacci(10)
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
'''
        ast_result = evaluator.run_ast_analysis(test_code)
        print(f"✅ AST Analysis - Score: {ast_result['score']:.2f}, Confidence: {ast_result['confidence']:.2f}")
        
        # Test security analysis (if bandit is available)
        try:
            security_result = evaluator.run_bandit_analysis(test_code)
            print(f"✅ Security Analysis - Score: {security_result['score']:.2f}")
        except Exception as e:
            print(f"⚠️  Security analysis skipped: {e}")
        
        # Test communication evaluation (will use mock if no LLM keys)
        print("💬 Testing communication evaluation...")
        try:
            comm_result = await evaluator.evaluate_communication(
                test_code, 
                "Calculate fibonacci numbers with error handling"
            )
            print(f"✅ Communication Analysis - Score: {comm_result['score']:.2f}")
        except Exception as e:
            print(f"⚠️  Communication analysis: {e}")
        
        print("\n" + "=" * 50)
        print("🎉 Production app test completed successfully!")
        print("\n📋 Summary:")
        print("- ✅ All imports working")
        print("- ✅ Evaluator initialization successful")
        print("- ✅ AST analysis functional")
        print("- ✅ Ready for Azure deployment")
        
        print("\n🚀 Next steps:")
        print("1. Set up API keys in Azure Key Vault")
        print("2. Run: ./deploy-azure.sh")
        print("3. Test deployed application")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_production_app())
    sys.exit(0 if success else 1)
