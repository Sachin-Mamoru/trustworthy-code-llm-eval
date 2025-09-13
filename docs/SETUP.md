# TrustworthyCodeLLM Setup Guide

## Prerequisites

- Python 3.8+
- Node.js 16+ (for web dashboard)
- Git
- 8GB+ RAM recommended

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/trustworthy-code-llm-eval.git
cd trustworthy-code-llm-eval
```

### 2. Python Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install package in development mode
pip install -e .
```

### 4. Configuration

```bash
# Copy configuration template
cp config/config.template.yaml config/config.yaml

# Edit configuration as needed
nano config/config.yaml
```

### 5. Web Dashboard Setup (Optional)

```bash
cd web_dashboard
npm install
npm run build
cd ..
```

## Quick Start

### Basic Evaluation

```python
from src.framework import MultiModalEvaluationFramework, CodeSample
from src.evaluators.enhanced_communication import EnhancedCommunicationEvaluator

# Initialize framework
framework = MultiModalEvaluationFramework()
framework.register_evaluator(EnhancedCommunicationEvaluator())

# Create test sample
sample = CodeSample(
    id="test_001",
    source_code="def greet(name): return f'Hello, {name}!'",
    problem_description="Create a greeting function"
)

# Run evaluation
results = await framework.evaluate_code_sample(sample)
```

### Running Examples

```bash
# Basic evaluation example
python examples/basic_evaluation.py

# Comprehensive evaluation with all modalities
python examples/comprehensive_evaluation.py

# Benchmark comparison with HumanEvalComm
python examples/benchmark_comparison.py
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src/

# Run specific test category
pytest tests/test_evaluators/
```

## Configuration Options

### Core Framework Settings

```yaml
framework:
  max_concurrent_evaluations: 10
  timeout_seconds: 30
  cache_results: true
  
evaluation:
  security:
    enabled: true
    strict_mode: false
  communication:
    llm_models: ["gpt-4", "claude-3"]
    ensemble_size: 3
  execution:
    sandbox_enabled: true
    resource_limits:
      memory_mb: 512
      cpu_time_seconds: 10
```

### Security Configuration

```yaml
security:
  sandbox:
    network_access: false
    file_system_access: "read_only"
    allowed_imports:
      - "math"
      - "statistics"
      - "itertools"
    blocked_imports:
      - "os"
      - "subprocess"
      - "socket"
```

## Environment Variables

```bash
# Optional: OpenAI API key for LLM evaluation
export OPENAI_API_KEY="your-api-key"

# Optional: HuggingFace token for model access
export HF_TOKEN="your-hf-token"

# Optional: Enable debug logging
export TRUSTWORTHY_DEBUG=1
```

## Troubleshooting

### Common Issues

**Import Errors:**
```bash
# Ensure package is installed in development mode
pip install -e .
```

**Permission Errors:**
```bash
# On macOS/Linux, ensure proper permissions
chmod +x scripts/*.sh
```

**Memory Issues:**
```bash
# Reduce concurrent evaluations
export MAX_CONCURRENT_EVALS=5
```

### Performance Optimization

1. **Enable Result Caching:**
   - Set `cache_results: true` in configuration
   - Cache stored in `~/.cache/trustworthy-code-llm/`

2. **Parallel Evaluation:**
   - Adjust `max_concurrent_evaluations` based on system capabilities
   - Monitor memory usage during evaluation

3. **Resource Limits:**
   - Configure appropriate timeouts for your hardware
   - Use strict sandboxing for better isolation

## Development

### Adding New Evaluators

1. Inherit from `TrustworthinessEvaluator`
2. Implement required abstract methods
3. Add tests in `tests/test_evaluators/`
4. Register in framework initialization

### Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit pull request

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

## Support

- **Documentation:** [docs/](docs/)
- **Issues:** GitHub Issues
- **Discussions:** GitHub Discussions
- **Email:** [maintainer-email]

## Docker Setup (Advanced)

```bash
# Build Docker image
docker build -t trustworthy-code-llm .

# Run evaluation in container
docker run -v $(pwd)/data:/app/data trustworthy-code-llm python examples/comprehensive_evaluation.py
```

## Cloud Deployment

See [docs/deployment.md](deployment.md) for cloud deployment instructions including:
- AWS Lambda deployment
- Google Cloud Functions
- Azure Container Instances
- Kubernetes configurations
