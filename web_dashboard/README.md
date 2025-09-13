# TrustworthyCodeLLM Evaluation Framework

A comprehensive, production-ready framework for evaluating Python code quality using hybrid analysis methods that combine static code analysis with AI-powered assessment.

## Features

- **Dual Analysis Engine**: Combines AST parsing with OpenAI GPT-3.5-turbo evaluation
- **Real-time Web Interface**: Clean, professional dashboard for code submission and analysis
- **Production Deployment**: Ready for Azure Container Apps with full CI/CD support
- **Comprehensive Scoring**: Weighted scoring system providing actionable feedback
- **Research Foundation**: Built on academic research principles for trustworthy code evaluation

## Live Demo

🌐 **Production Application**: https://trustworthy-code-llm-app.calmtree-8794f23a.eastus.azurecontainerapps.io

Try it with any Python code to see real-time analysis results.

## Quick Start

### Local Development

1. **Setup Environment**:
```bash
cd web_dashboard
python -m venv venv
source venv/bin/activate
pip install -r requirements-llm.txt
```

2. **Configure OpenAI**:
```bash
cp .env.template .env
# Add your OPENAI_API_KEY to .env
```

3. **Run Application**:
```bash
python llm_app.py
```

4. **Access Dashboard**: http://localhost:3003

### Azure Deployment

Deploy to Azure Container Apps in one command:

```bash
chmod +x deploy-azure-llm.sh
./deploy-azure-llm.sh
```

## How It Works

### Analysis Methods

**AST Analysis (40% weight)**
- Structural code parsing
- Complexity metrics calculation
- Syntax validation
- Quantitative assessment

**LLM Analysis (60% weight)**
- Code logic evaluation
- Style and readability assessment
- Documentation quality review
- Best practices verification

### Scoring System

The framework provides scores on a 0-100% scale:
- **80-100%**: Production-ready code with excellent practices
- **60-79%**: Good quality code with minor improvements needed
- **40-59%**: Functional code requiring significant improvements
- **0-39%**: Code with major issues requiring substantial revision

## Project Structure

```
├── web_dashboard/           # Production web application
│   ├── llm_app.py          # Main FastAPI application
│   ├── requirements-llm.txt # Python dependencies
│   ├── Dockerfile-llm      # Container configuration
│   └── deploy-azure-llm.sh # Azure deployment script
├── src/                    # Core evaluation framework
│   ├── framework.py        # Main evaluation framework
│   └── evaluators/         # Specialized evaluators
├── notebooks/              # Research and analysis notebooks
├── tests/                  # Test suites
└── docs/                   # Documentation
```

## Technology Stack

- **Framework**: FastAPI for high-performance web services
- **Analysis**: Python AST parsing for structural analysis
- **AI Integration**: OpenAI GPT-3.5-turbo for intelligent evaluation
- **Deployment**: Docker containers on Azure Container Apps
- **Frontend**: Modern HTML/CSS/JavaScript interface

## API Endpoints

- `GET /` - Main dashboard interface
- `POST /evaluate` - Code evaluation endpoint
- `GET /health` - Application health check

## Configuration

### Environment Variables

- `OPENAI_API_KEY` - Required for AI-powered analysis
- `PORT` - Application port (default: 3003)

### OpenAI Settings

- Model: gpt-3.5-turbo
- Temperature: 0.3 (for consistent analysis)
- Max tokens: 1000 per evaluation

## Research Applications

This framework supports academic research in:
- Code quality assessment methodologies
- AI-assisted code evaluation
- Trustworthiness metrics for generated code
- Hybrid analysis approaches

## Security & Performance

- **Secure Processing**: No code execution, static analysis only
- **Input Validation**: Comprehensive sanitization of user inputs
- **Resource Efficiency**: Optimized for production workloads
- **Error Handling**: Robust fallback mechanisms

## Contributing

This project follows standard development practices:

1. Fork the repository
2. Create feature branches
3. Submit pull requests
4. Ensure tests pass

## License

This project is developed for research and educational purposes.

## Contact

For questions about deployment, usage, or research applications, please refer to the documentation or create an issue in the repository.
