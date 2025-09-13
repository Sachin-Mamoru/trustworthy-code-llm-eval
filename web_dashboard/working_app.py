#!/usr/bin/env python3
"""
Working production app - completely rewritten to fix form submission
"""

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import ast
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="TrustworthyCodeLLM - Working", version="1.0.0")

class CodeEvaluator:
    def evaluate_code(self, code: str, problem_description: str) -> dict:
        try:
            tree = ast.parse(code)
            functions = len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)])
            classes = len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)])
            lines = len([l for l in code.split('\n') if l.strip()])
            complexity = len([n for n in ast.walk(tree) if isinstance(n, (ast.If, ast.For, ast.While))])
            
            structure_score = min(1.0, (functions + classes) / max(1, lines / 10))
            
            # Communication analysis
            comment_lines = len([l for l in code.split('\n') if l.strip().startswith('#')])
            docstring_count = code.count('"""') + code.count("'''")
            comm_score = min(1.0, (comment_lines + docstring_count/2) / max(1, lines / 5))
            
            overall_score = (structure_score + comm_score) / 2
            
            return {
                "overall_score": overall_score,
                "structure_analysis": {
                    "score": structure_score,
                    "metrics": {"functions": functions, "classes": classes, "lines": lines, "complexity": complexity},
                    "feedback": f"Code contains {functions} functions, {classes} classes in {lines} lines"
                },
                "communication_analysis": {
                    "score": comm_score,
                    "metrics": {"comment_lines": comment_lines, "docstrings": docstring_count//2, "total_lines": lines},
                    "feedback": f"Communication analysis: {comment_lines} comments, {docstring_count//2} docstrings"
                },
                "summary": f"Overall evaluation score: {overall_score:.2f}/1.0"
            }
            
        except SyntaxError as e:
            return {
                "overall_score": 0.0,
                "structure_analysis": {"score": 0.0, "error": f"Syntax Error: {str(e)}", "feedback": "Code has syntax errors"},
                "communication_analysis": {"score": 0.0, "feedback": "Cannot analyze due to syntax errors"},
                "summary": "Code evaluation failed due to syntax errors"
            }

evaluator = CodeEvaluator()

@app.get("/", response_class=HTMLResponse)
async def home():
    html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TrustworthyCodeLLM - Working Version</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
        .container { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #333; text-align: center; margin-bottom: 30px; }
        .form-group { margin-bottom: 20px; }
        label { display: block; margin-bottom: 8px; font-weight: bold; color: #555; }
        input, textarea { width: 100%; padding: 12px; border: 2px solid #ddd; border-radius: 6px; font-size: 14px; box-sizing: border-box; }
        input:focus, textarea:focus { border-color: #007bff; outline: none; }
        button { width: 100%; padding: 15px; background: #007bff; color: white; border: none; border-radius: 6px; font-size: 16px; font-weight: bold; cursor: pointer; }
        button:hover { background: #0056b3; }
        button:disabled { background: #ccc; cursor: not-allowed; }
        .results { margin-top: 30px; padding: 20px; background: #f8f9fa; border-radius: 6px; border-left: 4px solid #007bff; }
        .score { font-size: 18px; font-weight: bold; margin: 15px 0; padding: 10px; border-radius: 6px; text-align: center; }
        .excellent { background: #d4edda; color: #155724; }
        .good { background: #d1ecf1; color: #0c5460; }
        .fair { background: #fff3cd; color: #856404; }
        .poor { background: #f8d7da; color: #721c24; }
        .error { background: #f8d7da; color: #721c24; padding: 15px; border-radius: 6px; margin: 10px 0; }
        .loading { text-align: center; padding: 20px; color: #666; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔒 TrustworthyCodeLLM - Working Version</h1>
        
        <form id="codeForm">
            <div class="form-group">
                <label for="problem">Problem Description:</label>
                <input type="text" id="problem" name="problem_description" placeholder="Describe the coding problem..." required>
            </div>
            
            <div class="form-group">
                <label for="code">Python Code:</label>
                <textarea id="code" name="code" rows="15" placeholder="Enter your Python code here..." required></textarea>
            </div>
            
            <button type="submit" id="submitButton">🚀 Evaluate Code</button>
        </form>
        
        <div id="results" style="display: none;"></div>
    </div>
    
    <script>
        // Wait for DOM to be ready
        document.addEventListener('DOMContentLoaded', function() {
            const form = document.getElementById('codeForm');
            const button = document.getElementById('submitButton');
            const results = document.getElementById('results');
            
            // Add form submit handler
            form.addEventListener('submit', function(event) {
                // Prevent default form submission
                event.preventDefault();
                event.stopPropagation();
                
                console.log('Form submitted - JavaScript working!');
                
                // Get form data
                const formData = new FormData(form);
                const code = formData.get('code');
                const problem = formData.get('problem_description');
                
                console.log('Code length:', code ? code.length : 0);
                console.log('Problem:', problem);
                
                if (!code || !problem) {
                    alert('Please fill in both fields');
                    return;
                }
                
                // Show loading
                button.disabled = true;
                button.textContent = '⏳ Analyzing...';
                results.style.display = 'block';
                results.innerHTML = '<div class="loading">🔍 Running analysis...</div>';
                
                // Send request
                fetch('/evaluate', {
                    method: 'POST',
                    body: formData
                })
                .then(response => {
                    console.log('Response status:', response.status);
                    if (!response.ok) {
                        throw new Error('Network response was not ok');
                    }
                    return response.json();
                })
                .then(data => {
                    console.log('Success:', data);
                    displayResults(data);
                })
                .catch(error => {
                    console.error('Error:', error);
                    results.innerHTML = '<div class="error">❌ Error: ' + error.message + '</div>';
                })
                .finally(() => {
                    button.disabled = false;
                    button.textContent = '🚀 Evaluate Code';
                });
            });
            
            function displayResults(data) {
                let html = '';
                
                if (data.overall_score !== undefined) {
                    const score = data.overall_score;
                    let scoreClass = 'poor';
                    if (score >= 0.8) scoreClass = 'excellent';
                    else if (score >= 0.6) scoreClass = 'good';
                    else if (score >= 0.4) scoreClass = 'fair';
                    
                    html += '<div class="score ' + scoreClass + '">Overall Score: ' + (score * 100).toFixed(1) + '%</div>';
                    
                    if (data.structure_analysis) {
                        html += '<h4>📊 Code Structure</h4>';
                        html += '<p>Score: ' + (data.structure_analysis.score * 100).toFixed(1) + '%</p>';
                        html += '<p>' + data.structure_analysis.feedback + '</p>';
                    }
                    
                    if (data.communication_analysis) {
                        html += '<h4>💬 Communication</h4>';
                        html += '<p>Score: ' + (data.communication_analysis.score * 100).toFixed(1) + '%</p>';
                        html += '<p>' + data.communication_analysis.feedback + '</p>';
                    }
                    
                    html += '<div style="margin-top: 20px; padding: 15px; background: #e7f3ff; border-radius: 6px;">';
                    html += '<strong>Summary:</strong> ' + data.summary;
                    html += '</div>';
                }
                
                results.innerHTML = html;
            }
        });
    </script>
</body>
</html>'''
    
    return HTMLResponse(content=html)

@app.get("/health")
async def health():
    return {"status": "healthy", "version": "working"}

@app.post("/evaluate")
async def evaluate_code(
    code: str = Form(...),
    problem_description: str = Form(...)
):
    logger.info(f"POST request received - Code: {len(code)} chars, Problem: {problem_description[:50]}...")
    
    try:
        results = evaluator.evaluate_code(code, problem_description)
        logger.info(f"Evaluation successful: {results['overall_score']:.2f}")
        return JSONResponse(content=results)
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 3002))
    uvicorn.run(
        "working_app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )
