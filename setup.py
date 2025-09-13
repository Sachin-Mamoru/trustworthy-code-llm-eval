#!/usr/bin/env python3
"""
Setup script for TrustworthyCodeLLM evaluation framework.

This script sets up the development environment and validates the installation.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        # Use the correct Python path
        command = command.replace("python ", "/opt/homebrew/bin/python3 ")
        command = command.replace("pip ", "/opt/homebrew/bin/python3 -m pip ")
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return None


def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} is not supported. Please use Python 3.8+")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def setup_virtual_environment():
    """Set up and activate virtual environment."""
    venv_path = Path("venv")
    
    if not venv_path.exists():
        print("🔄 Creating virtual environment...")
        result = run_command("/opt/homebrew/bin/python3 -m venv venv", "Virtual environment creation")
        if result is None:
            return False
    else:
        print("✅ Virtual environment already exists")
    
    # Check if we're in the virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment is active")
        return True
    else:
        print("⚠️  Virtual environment exists but is not active")
        print("   Please activate it with:")
        if os.name == 'nt':  # Windows
            print("   venv\\Scripts\\activate")
        else:  # Unix/Linux/macOS
            print("   source venv/bin/activate")
        return True


def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    
    # Upgrade pip first
    run_command("/opt/homebrew/bin/python3 -m pip install --upgrade pip", "Pip upgrade")
    
    # Install requirements
    result = run_command("/opt/homebrew/bin/python3 -m pip install -r requirements.txt", "Requirements installation")
    if result is None:
        print("⚠️  Some dependencies may have failed to install")
        print("   You can install them manually with: pip install -r requirements.txt")
        return False
    
    return True


def create_directory_structure():
    """Create necessary directories."""
    print("📁 Creating directory structure...")
    
    directories = [
        "data",
        "results",
        "logs",
        "config",
        "docs",
        "scripts",
        "notebooks"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created/verified directory: {directory}")


def create_config_files():
    """Create basic configuration files."""
    print("⚙️  Creating configuration files...")
    
    # Create .gitignore
    gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Project specific
logs/
results/
data/cache/
*.log
.env
config/secrets.yaml

# Jupyter
.ipynb_checkpoints/
    """
    
    with open(".gitignore", "w") as f:
        f.write(gitignore_content.strip())
    
    # Create basic logging config
    logging_config = """
version: 1
disable_existing_loggers: False

formatters:
  standard:
    format: '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
  detailed:
    format: '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s'

handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: standard
    stream: ext://sys.stdout
  
  file:
    class: logging.FileHandler
    level: DEBUG
    formatter: detailed
    filename: logs/framework.log
    mode: a

loggers:
  src:
    level: DEBUG
    handlers: [console, file]
    propagate: False

root:
  level: INFO
  handlers: [console]
    """
    
    config_dir = Path("config")
    with open(config_dir / "logging.yaml", "w") as f:
        f.write(logging_config.strip())
    
    print("✅ Configuration files created")


def validate_installation():
    """Validate that the framework can be imported."""
    print("🔍 Validating installation...")
    
    try:
        # Try to import core modules
        sys.path.insert(0, str(Path.cwd()))
        
        from src.framework import MultiModalEvaluationFramework, CodeSample
        from src.evaluators.enhanced_communication import EnhancedCommunicationEvaluator
        
        # Create a simple test
        framework = MultiModalEvaluationFramework()
        evaluator = EnhancedCommunicationEvaluator()
        framework.register_evaluator(evaluator)
        
        sample = CodeSample(
            id="test",
            source_code="def hello(): return 'Hello, World!'",
            problem_description="Test function",
            language="python"
        )
        
        # This should work without errors
        result = evaluator.evaluate(sample)
        
        print("✅ Framework validation successful")
        print(f"   Sample evaluation score: {result.score:.2f}")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Some dependencies may be missing")
        return False
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False


def run_sample_tests():
    """Run a few sample tests to verify functionality."""
    print("🧪 Running sample tests...")
    
    test_result = run_command("/opt/homebrew/bin/python3 -m pytest tests/ -v --tb=short", "Sample tests")
    if test_result is None:
        print("⚠️  Some tests may have failed, but this is normal during initial setup")
        return True
    
    return True


def main():
    """Main setup function."""
    print("🚀 TrustworthyCodeLLM Framework Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Setup virtual environment
    setup_virtual_environment()
    
    # Install dependencies
    if not install_dependencies():
        print("⚠️  Continuing with setup despite dependency installation issues...")
    
    # Create directory structure
    create_directory_structure()
    
    # Create config files
    create_config_files()
    
    # Validate installation
    if validate_installation():
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Activate virtual environment (if not already active)")
        print("2. Run the example: python examples/comprehensive_evaluation.py")
        print("3. Run tests: python -m pytest tests/")
        print("4. Start developing your evaluations!")
    else:
        print("\n⚠️  Setup completed with some issues")
        print("Please check the error messages above and resolve any dependency issues")
    
    # Run sample tests
    run_sample_tests()


if __name__ == "__main__":
    main()
