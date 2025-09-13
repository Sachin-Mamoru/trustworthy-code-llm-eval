#!/usr/bin/env python3
"""
Debug version of the production app to fix form submission issues.
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

app = FastAPI(title="TrustworthyCodeLLM Debug", version="1.0.0")

class SimpleEvaluator:
    """Simple code evaluator for debugging."""
    
    def evaluate_code(self, code: str, problem_description: str) -> dict:
        """Evaluate code and return results."""
        try:
            tree = ast.parse(code)
            
            # Count elements
            functions = len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)])
            classes = len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)])
            lines = len([l for l in code.split('\n') if l.strip()])
            
            # Simple scoring
            score = min(1.0, (functions + classes) / max(1, lines / 10))
            
            return {
                "success": True,
                "score": score,
                "functions": functions,
                "classes": classes,
                "lines": lines,
                "feedback": f"Found {functions} functions, {classes} classes in {lines} lines"
            }
            
        except SyntaxError as e:
            return {
                "success": False,
                "error": f"Syntax Error: {str(e)}",
                "score": 0.0
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error: {str(e)}",
                "score": 0.0
            }

evaluator = SimpleEvaluator()

@app.get("/", response_class=HTMLResponse)
async def home():
    """Home page with simple form."""
    html = '''
<!DOCTYPE html>
<html>
<head>
    <title>Debug TrustworthyCodeLLM</title>
    <style>
        body { font-family: Arial; margin: 40px; }
        textarea { width: 100%; height: 200px; margin: 10px 0; }
        input[type="text"] { width: 100%; margin: 10px 0; padding: 10px; }
        button { padding: 15px 30px; background: #007bff; color: white; border: none; font-size: 16px; }
        .results { margin-top: 20px; padding: 20px; background: #f0f0f0; }
        .error { color: red; }
        .success { color: green; }
    </style>
</head>
<body>
    <h1>🔒 TrustworthyCodeLLM - Debug Version</h1>
    
    <form id="evalForm">
        <div>
            <label>Problem Description:</label>
            <input type="text" name="problem_description" placeholder="Describe the problem..." required>
        </div>
        
        <div>
            <label>Python Code:</label>
            <textarea name="code" placeholder="Enter your Python code here..." required></textarea>
        </div>
        
        <button type="submit">🚀 Evaluate Code</button>
    </form>
    
    <div id="results" style="display: none;"></div>
    
    <script>
        document.getElementById('evalForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const results = document.getElementById('results');
            
            console.log('Submitting form...');
            console.log('Problem:', formData.get('problem_description'));
            console.log('Code:', formData.get('code'));
            
            results.style.display = 'block';
            results.innerHTML = '<p>⏳ Analyzing...</p>';
            
            try {
                const response = await fetch('/evaluate', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                console.log('Response:', data);
                
                if (response.ok) {
                    if (data.success) {
                        results.innerHTML = `
                            <div class="results success">
                                <h3>✅ Analysis Complete</h3>
                                <p><strong>Score:</strong> ${(data.score * 100).toFixed(1)}%</p>
                                <p><strong>Functions:</strong> ${data.functions}</p>
                                <p><strong>Classes:</strong> ${data.classes}</p>
                                <p><strong>Lines:</strong> ${data.lines}</p>
                                <p><strong>Feedback:</strong> ${data.feedback}</p>
                            </div>
                        `;
                    } else {
                        results.innerHTML = `
                            <div class="results error">
                                <h3>❌ Analysis Failed</h3>
                                <p><strong>Error:</strong> ${data.error}</p>
                                <p><strong>Score:</strong> ${(data.score * 100).toFixed(1)}%</p>
                            </div>
                        `;
                    }
                } else {
                    results.innerHTML = `<div class="error">❌ Server Error: ${data.detail}</div>`;
                }
            } catch (error) {
                console.error('Error:', error);
                results.innerHTML = `<div class="error">❌ Network Error: ${error.message}</div>`;
            }
        });
    </script>
</body>
</html>
    '''
    return HTMLResponse(content=html)

@app.get("/health")
async def health():
    return {"status": "healthy", "debug": True}

@app.post("/evaluate")
async def evaluate_code(
    code: str = Form(...),
    problem_description: str = Form(...)
):
    """Evaluate code submission."""
    logger.info(f"Received evaluation request: {len(code)} chars, problem: {problem_description[:50]}...")
    
    try:
        results = evaluator.evaluate_code(code, problem_description)
        logger.info(f"Evaluation result: {results}")
        return JSONResponse(content=results)
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 3001))
    uvicorn.run(
        "debug_app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )
