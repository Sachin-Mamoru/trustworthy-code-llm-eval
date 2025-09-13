#!/usr/bin/env python3
"""
Simple production app for Azure deployment.
Ready for supervisor testing with real evaluation capabilities.
"""

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
import ast
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="TrustworthyCodeLLM - Production Ready", version="2.0.0")

# Setup templates
templates_dir = Path(__file__).parent / "templates"
templates_dir.mkdir(exist_ok=True)
templates = Jinja2Templates(directory=str(templates_dir))

class SimpleEvaluator:
    """Simple but effective code evaluator for production use."""
    
    def evaluate_code_structure(self, code: str) -> dict:
        """Evaluate code structure using AST analysis."""
        try:
            tree = ast.parse(code)
            
            # Count code elements
            functions = len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)])
            classes = len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)])
            imports = len([n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))])
            
            # Calculate metrics
            lines = len([l for l in code.split('\n') if l.strip()])
            complexity = len([n for n in ast.walk(tree) if isinstance(n, (ast.If, ast.For, ast.While))])
            
            # Score calculation
            if lines == 0:
                score = 0.0
            else:
                structure_score = min(1.0, (functions + classes) / max(1, lines / 15))
                organization_score = min(1.0, imports / max(1, functions + classes))
                complexity_score = min(1.0, complexity / max(1, functions))
                score = (structure_score + organization_score + complexity_score) / 3
            
            return {
                "score": score,
                "metrics": {
                    "functions": functions,
                    "classes": classes,
                    "imports": imports,
                    "lines": lines,
                    "complexity": complexity
                },
                "feedback": f"Code contains {functions} functions, {classes} classes with complexity score {complexity}"
            }
            
        except SyntaxError as e:
            return {
                "score": 0.0,
                "error": f"Syntax Error: {str(e)}",
                "feedback": "Code has syntax errors that prevent analysis"
            }
        except Exception as e:
            return {
                "score": 0.3,
                "error": f"Analysis Error: {str(e)}",
                "feedback": "Could not complete code structure analysis"
            }
    
    def evaluate_communication(self, code: str, problem_description: str) -> dict:
        """Evaluate communication aspects of the code."""
        try:
            # Basic heuristics for communication quality
            lines = code.split('\n')
            total_lines = len(lines)
            
            # Count comments
            comment_lines = len([l for l in lines if l.strip().startswith('#')])
            
            # Count docstrings (rough approximation)
            docstring_count = code.count('"""') + code.count("'''")
            
            # Check for meaningful variable names (avoid single letters except common ones)
            import re
            variables = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', code)
            meaningful_vars = len([v for v in set(variables) 
                                 if len(v) > 2 and v not in ['for', 'if', 'in', 'def', 'class']])
            
            # Score calculation
            comment_ratio = comment_lines / max(1, total_lines)
            docstring_score = min(1.0, docstring_count / 4)  # Assume good docs have multiple docstrings
            naming_score = min(1.0, meaningful_vars / max(1, total_lines / 5))
            
            communication_score = (comment_ratio + docstring_score + naming_score) / 3
            
            return {
                "score": communication_score,
                "metrics": {
                    "comment_lines": comment_lines,
                    "docstrings": docstring_count // 2,  # Divide by 2 for pairs
                    "meaningful_variables": meaningful_vars,
                    "total_lines": total_lines
                },
                "feedback": f"Communication analysis: {comment_lines} comments, {docstring_count//2} docstrings, {meaningful_vars} meaningful variables"
            }
            
        except Exception as e:
            return {
                "score": 0.5,
                "error": str(e),
                "feedback": "Could not complete communication analysis"
            }
    
    def comprehensive_evaluation(self, code: str, problem_description: str) -> dict:
        """Run comprehensive evaluation."""
        logger.info("Starting comprehensive evaluation")
        
        # Run individual analyses
        structure_result = self.evaluate_code_structure(code)
        communication_result = self.evaluate_communication(code, problem_description)
        
        # Calculate overall score
        structure_weight = 0.6
        communication_weight = 0.4
        
        overall_score = (
            structure_result["score"] * structure_weight +
            communication_result["score"] * communication_weight
        )
        
        return {
            "overall_score": overall_score,
            "structure_analysis": structure_result,
            "communication_analysis": communication_result,
            "summary": f"Overall evaluation score: {overall_score:.2f}/1.0"
        }

# Global evaluator instance
evaluator = SimpleEvaluator()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with evaluation interface."""
    return templates.TemplateResponse("evaluation.html", {"request": request})

@app.get("/health")
async def health_check():
    """Health check for Azure deployment."""
    return {
        "status": "healthy",
        "service": "TrustworthyCodeLLM Production",
        "version": "2.0.0",
        "ready_for_evaluation": True
    }

@app.post("/evaluate")
async def evaluate_code(
    code: str = Form(...),
    problem_description: str = Form(...)
):
    """Evaluate submitted code."""
    try:
        logger.info(f"Evaluating code submission: {len(code)} characters")
        results = evaluator.comprehensive_evaluation(code, problem_description)
        return JSONResponse(content=results)
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Create the HTML template
def create_evaluation_template():
    """Create the evaluation interface template."""
    template_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TrustworthyCodeLLM - Production Evaluation</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            margin-bottom: 40px;
            color: #333;
        }
        .header h1 {
            color: #667eea;
            margin-bottom: 10px;
            font-size: 2.5em;
        }
        .header p {
            color: #666;
            font-size: 1.1em;
        }
        .form-group {
            margin-bottom: 25px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #333;
            font-size: 1.1em;
        }
        textarea, input {
            width: 100%;
            padding: 15px;
            border: 2px solid #e1e5e9;
            border-radius: 8px;
            font-size: 14px;
            font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
            transition: border-color 0.3s ease;
            box-sizing: border-box;
        }
        textarea:focus, input:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 30px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1.1em;
            font-weight: 600;
            transition: transform 0.2s ease;
            width: 100%;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        button:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }
        .results {
            margin-top: 30px;
            padding: 25px;
            background: #f8f9fa;
            border-radius: 10px;
            border-left: 5px solid #667eea;
        }
        .score {
            font-size: 1.4em;
            font-weight: bold;
            margin: 15px 0;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }
        .score.excellent { background: #d4edda; color: #155724; border: 2px solid #c3e6cb; }
        .score.good { background: #d1ecf1; color: #0c5460; border: 2px solid #bee5eb; }
        .score.fair { background: #fff3cd; color: #856404; border: 2px solid #ffeaa7; }
        .score.poor { background: #f8d7da; color: #721c24; border: 2px solid #f5c6cb; }
        .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .metric-card {
            background: white;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #dee2e6;
        }
        .metric-title {
            font-weight: 600;
            color: #667eea;
            margin-bottom: 8px;
        }
        .metric-value {
            font-size: 1.2em;
            font-weight: bold;
            color: #333;
        }
        .loading {
            text-align: center;
            padding: 40px;
            color: #666;
        }
        .error {
            background: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #f5c6cb;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔒 TrustworthyCodeLLM</h1>
            <p>Production-Ready Code Evaluation Platform</p>
            <p><small>Real-time analysis for your supervisor's review</small></p>
        </div>
        
        <form id="evaluationForm">
            <div class="form-group">
                <label for="problem">📋 Problem Description:</label>
                <textarea id="problem" name="problem_description" rows="4" 
                         placeholder="Describe the coding problem or task you're solving..." required></textarea>
            </div>
            
            <div class="form-group">
                <label for="code">💻 Python Code to Evaluate:</label>
                <textarea id="code" name="code" rows="20" 
                         placeholder="Enter your Python code here..." required></textarea>
            </div>
            
            <button type="submit" id="submitBtn">🚀 Evaluate Code Quality</button>
        </form>
        
        <div id="results" class="results" style="display: none;">
            <h3>📊 Evaluation Results</h3>
            <div id="resultsContent"></div>
        </div>
    </div>
    
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const form = document.getElementById('evaluationForm');
            const button = document.getElementById('submitBtn');
            const results = document.getElementById('results');
            const content = document.getElementById('resultsContent');
            
            form.addEventListener('submit', async function(e) {
                e.preventDefault();
                e.stopPropagation();
                
                console.log('Form submission intercepted');
                
                const formData = new FormData(form);
                const codeValue = formData.get('code');
                const problemValue = formData.get('problem_description');
                
                console.log('Code length:', codeValue ? codeValue.length : 0);
                console.log('Problem:', problemValue);
                
                if (!codeValue || !problemValue) {
                    alert('Please fill in both fields');
                    return;
                }
                
                // Show loading state
                button.disabled = true;
                button.textContent = '⏳ Analyzing Code...';
                results.style.display = 'block';
                content.innerHTML = '<div class="loading">🔍 Running comprehensive analysis...</div>';
                
                try {
                    console.log('Sending POST request...');
                    const response = await fetch('/evaluate', {
                        method: 'POST',
                        body: formData
                    });
                    
                    console.log('Response status:', response.status);
                    
                    if (!response.ok) {
                        throw new Error(`HTTP ${response.status}`);
                    }
                    
                    const data = await response.json();
                    console.log('Response data:', data);
                
                if (response.ok) {
                    let html = '';
                    
                    // Overall score
                    const score = data.overall_score;
                    let scoreClass = 'poor';
                    if (score >= 0.8) scoreClass = 'excellent';
                    else if (score >= 0.6) scoreClass = 'good';
                    else if (score >= 0.4) scoreClass = 'fair';
                    
                    html += `<div class="score ${scoreClass}">
                        Overall Score: ${(score * 100).toFixed(1)}% 
                        ${score >= 0.8 ? '🌟' : score >= 0.6 ? '👍' : score >= 0.4 ? '⚠️' : '❌'}
                    </div>`;
                    
                    // Structure Analysis
                    if (data.structure_analysis) {
                        const struct = data.structure_analysis;
                        html += '<h4>📊 Code Structure Analysis</h4>';
                        html += `<p><strong>Score:</strong> ${(struct.score * 100).toFixed(1)}%</p>`;
                        html += `<p>${struct.feedback}</p>`;
                        
                        if (struct.metrics) {
                            html += '<div class="metrics">';
                            html += `<div class="metric-card">
                                <div class="metric-title">Functions</div>
                                <div class="metric-value">${struct.metrics.functions}</div>
                            </div>`;
                            html += `<div class="metric-card">
                                <div class="metric-title">Classes</div>
                                <div class="metric-value">${struct.metrics.classes}</div>
                            </div>`;
                            html += `<div class="metric-card">
                                <div class="metric-title">Lines of Code</div>
                                <div class="metric-value">${struct.metrics.lines}</div>
                            </div>`;
                            html += `<div class="metric-card">
                                <div class="metric-title">Complexity</div>
                                <div class="metric-value">${struct.metrics.complexity}</div>
                            </div>`;
                            html += '</div>';
                        }
                    }
                    
                    // Communication Analysis
                    if (data.communication_analysis) {
                        const comm = data.communication_analysis;
                        html += '<h4>💬 Communication Quality</h4>';
                        html += `<p><strong>Score:</strong> ${(comm.score * 100).toFixed(1)}%</p>`;
                        html += `<p>${comm.feedback}</p>`;
                        
                        if (comm.metrics) {
                            html += '<div class="metrics">';
                            html += `<div class="metric-card">
                                <div class="metric-title">Comments</div>
                                <div class="metric-value">${comm.metrics.comment_lines}</div>
                            </div>`;
                            html += `<div class="metric-card">
                                <div class="metric-title">Docstrings</div>
                                <div class="metric-value">${comm.metrics.docstrings}</div>
                            </div>`;
                            html += `<div class="metric-card">
                                <div class="metric-title">Meaningful Names</div>
                                <div class="metric-value">${comm.metrics.meaningful_variables}</div>
                            </div>`;
                            html += '</div>';
                        }
                    }
                    
                    html += `<div style="margin-top: 20px; padding: 15px; background: #e7f3ff; border-radius: 8px;">
                        <strong>Summary:</strong> ${data.summary}
                    </div>`;
                    
                    content.innerHTML = html;
                    
                    // Display results
                    let html = '';
                    
                    if (data.overall_score !== undefined) {
                        const score = data.overall_score;
                        let scoreClass = 'poor';
                        if (score >= 0.8) scoreClass = 'excellent';
                        else if (score >= 0.6) scoreClass = 'good';
                        else if (score >= 0.4) scoreClass = 'fair';
                        
                        html += `<div class="score ${scoreClass}">
                            Overall Score: ${(score * 100).toFixed(1)}% 
                            ${score >= 0.8 ? '🌟' : score >= 0.6 ? '👍' : score >= 0.4 ? '⚠️' : '❌'}
                        </div>`;
                        
                        // Structure analysis
                        if (data.structure_analysis) {
                            const struct = data.structure_analysis;
                            html += '<h4>📊 Code Structure Analysis</h4>';
                            html += `<p><strong>Score:</strong> ${(struct.score * 100).toFixed(1)}%</p>`;
                            html += `<p>${struct.feedback}</p>`;
                        }
                        
                        // Communication analysis
                        if (data.communication_analysis) {
                            const comm = data.communication_analysis;
                            html += '<h4>💬 Communication Quality</h4>';
                            html += `<p><strong>Score:</strong> ${(comm.score * 100).toFixed(1)}%</p>`;
                            html += `<p>${comm.feedback}</p>`;
                        }
                        
                        html += `<div style="margin-top: 20px; padding: 15px; background: #e7f3ff; border-radius: 8px;">
                            <strong>Summary:</strong> ${data.summary}
                        </div>`;
                    }
                    
                    content.innerHTML = html;
                    
                } catch (error) {
                    console.error('JavaScript error:', error);
                    content.innerHTML = `<div class="error">❌ Network error: ${error.message}</div>`;
                } finally {
                    button.disabled = false;
                    button.textContent = '🚀 Evaluate Code Quality';
                }
            });
        });
        
        // Add some sample code for testing
        document.addEventListener('DOMContentLoaded', function() {
            const sampleCode = `def fibonacci(n):
    """Calculate the nth Fibonacci number using recursion.
    
    Args:
        n (int): The position in the Fibonacci sequence
        
    Returns:
        int: The nth Fibonacci number
    """
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def main():
    """Main function to test fibonacci calculation."""
    try:
        # Calculate first 10 Fibonacci numbers
        for i in range(10):
            result = fibonacci(i)
            print(f"F({i}) = {result}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()`;
            
            document.getElementById('code').placeholder = 'Enter your Python code here...\n\nExample:\n' + sampleCode;
        });
    </script>
</body>
</html>'''
    
    templates_dir = Path(__file__).parent / "templates"
    templates_dir.mkdir(exist_ok=True)
    
    with open(templates_dir / "evaluation.html", "w") as f:
        f.write(template_content)

# Create template on startup
create_evaluation_template()

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "simple_production:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )
