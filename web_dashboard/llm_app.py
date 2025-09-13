#!/usr/bin/env python3
"""
LLM-Enhanced version with real OpenAI integration
"""

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import ast
import os
import logging
import asyncio
import json

# LLM Integration
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="TrustworthyCodeLLM - LLM Enhanced", version="2.0.0")

class LLMCodeEvaluator:
    def __init__(self):
        self.openai_available = False
        api_key = os.getenv("OPENAI_API_KEY")
        
        logger.info(f"OpenAI available: {OPENAI_AVAILABLE}")
        logger.info(f"API key present: {'Yes' if api_key else 'No'}")
        logger.info(f"API key length: {len(api_key) if api_key else 0}")
        
        if OPENAI_AVAILABLE and api_key:
            try:
                # Use old-style OpenAI API configuration
                openai.api_key = api_key
                self.openai_available = True
                logger.info("OpenAI API key configured successfully")
            except Exception as e:
                logger.error(f"Failed to configure OpenAI: {e}")
                logger.error(f"Error type: {type(e).__name__}")
                logger.info("Falling back to rule-based analysis only")
        else:
            if not OPENAI_AVAILABLE:
                logger.warning("OpenAI library not available")
            if not api_key:
                logger.warning("OpenAI API key not found in environment")
            logger.info("OpenAI not available - using rule-based analysis only")
    
    def ast_analysis(self, code: str) -> dict:
        """AST-based structural analysis"""
        try:
            tree = ast.parse(code)
            functions = len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)])
            classes = len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)])
            lines = len([l for l in code.split('\n') if l.strip()])
            complexity = len([n for n in ast.walk(tree) if isinstance(n, (ast.If, ast.For, ast.While))])
            
            structure_score = min(1.0, (functions + classes) / max(1, lines / 10))
            
            return {
                "score": structure_score,
                "metrics": {"functions": functions, "classes": classes, "lines": lines, "complexity": complexity},
                "feedback": f"AST Analysis: {functions} functions, {classes} classes, {complexity} control structures",
                "method": "AST Analysis"
            }
        except SyntaxError as e:
            return {
                "score": 0.0,
                "error": f"Syntax Error: {str(e)}",
                "feedback": "Code has syntax errors that prevent AST analysis",
                "method": "AST Analysis"
            }
    
    async def llm_analysis(self, code: str, problem_description: str) -> dict:
        """LLM-powered code quality analysis"""
        if not self.openai_available:
            return {
                "score": 0.5,
                "feedback": "LLM analysis not available (no API key configured)",
                "method": "Fallback - No LLM",
                "model": "None"
            }
        
        prompt = f"""
        Analyze this Python code for quality, readability, and correctness.
        
        Problem Description: {problem_description}
        
        Code:
        ```python
        {code}
        ```
        
        Evaluate the code on a scale of 0.0 to 1.0 based on:
        1. Code correctness and logic
        2. Readability and style
        3. Documentation quality
        4. Best practices adherence
        5. Problem-solving approach
        
        Respond with valid JSON only:
        {{
            "score": 0.0-1.0,
            "reasoning": "detailed explanation",
            "strengths": ["list of strengths"],
            "improvements": ["list of improvements"]
        }}
        """
        
        try:
            # Use chat completions API with gpt-3.5-turbo
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a code quality expert. Analyze code and respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                result = json.loads(result_text)
                return {
                    "score": float(result.get("score", 0.5)),
                    "feedback": result.get("reasoning", "LLM analysis completed"),
                    "strengths": result.get("strengths", []),
                    "improvements": result.get("improvements", []),
                    "method": "OpenAI GPT-3.5-turbo",
                    "model": "gpt-3.5-turbo"
                }
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                return {
                    "score": 0.7,
                    "feedback": result_text[:300] + "..." if len(result_text) > 300 else result_text,
                    "method": "OpenAI GPT-3.5-turbo (text)",
                    "model": "gpt-3.5-turbo"
                }
                
        except Exception as e:
            logger.error(f"LLM analysis error: {e}")
            return {
                "score": 0.5,
                "error": str(e),
                "feedback": f"LLM analysis failed: {str(e)}",
                "method": "Error fallback",
                "model": "None"
            }
    
    async def comprehensive_evaluation(self, code: str, problem_description: str) -> dict:
        """Run both AST and LLM analysis"""
        logger.info(f"Running comprehensive evaluation: {len(code)} chars")
        
        # Run AST analysis
        ast_result = self.ast_analysis(code)
        
        # Run LLM analysis
        llm_result = await self.llm_analysis(code, problem_description)
        
        # Calculate combined score
        ast_weight = 0.4
        llm_weight = 0.6
        
        combined_score = (ast_result["score"] * ast_weight + llm_result["score"] * llm_weight)
        
        return {
            "overall_score": combined_score,
            "ast_analysis": ast_result,
            "llm_analysis": llm_result,
            "evaluation_methods": {
                "ast": ast_result.get("method", "AST"),
                "llm": llm_result.get("method", "None"),
                "model": llm_result.get("model", "None")
            },
            "summary": f"Combined evaluation score: {combined_score:.2f}/1.0"
        }

evaluator = LLMCodeEvaluator()

@app.get("/", response_class=HTMLResponse)
async def home():
    openai_status = "✅ Connected" if evaluator.openai_available else "❌ No API Key"
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TrustworthyCodeLLM - LLM Enhanced</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
        .container {{ background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .status {{ background: #e7f3ff; padding: 10px; border-radius: 6px; margin-bottom: 20px; font-size: 14px; }}
        h1 {{ color: #333; text-align: center; margin-bottom: 20px; }}
        .form-group {{ margin-bottom: 20px; }}
        label {{ display: block; margin-bottom: 8px; font-weight: bold; color: #555; }}
        input, textarea {{ width: 100%; padding: 12px; border: 2px solid #ddd; border-radius: 6px; font-size: 14px; box-sizing: border-box; }}
        input:focus, textarea:focus {{ border-color: #007bff; outline: none; }}
        button {{ width: 100%; padding: 15px; background: #007bff; color: white; border: none; border-radius: 6px; font-size: 16px; font-weight: bold; cursor: pointer; }}
        button:hover {{ background: #0056b3; }}
        button:disabled {{ background: #ccc; cursor: not-allowed; }}
        .results {{ margin-top: 30px; padding: 20px; background: #f8f9fa; border-radius: 6px; border-left: 4px solid #007bff; }}
        .score {{ font-size: 18px; font-weight: bold; margin: 15px 0; padding: 10px; border-radius: 6px; text-align: center; }}
        .excellent {{ background: #d4edda; color: #155724; }}
        .good {{ background: #d1ecf1; color: #0c5460; }}
        .fair {{ background: #fff3cd; color: #856404; }}
        .poor {{ background: #f8d7da; color: #721c24; }}
        .analysis-section {{ margin: 20px 0; padding: 15px; background: white; border-radius: 6px; border: 1px solid #ddd; }}
        .method {{ font-size: 12px; color: #666; font-style: italic; }}
        .error {{ background: #f8d7da; color: #721c24; padding: 15px; border-radius: 6px; margin: 10px 0; }}
        .loading {{ text-align: center; padding: 20px; color: #666; }}
        .list {{ margin: 10px 0; }}
        .list li {{ margin: 5px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 TrustworthyCodeLLM - LLM Enhanced</h1>
        
        <div class="status">
            <strong>🔧 Analysis Methods:</strong> AST Analysis + OpenAI LLM<br>
            <strong>🤖 OpenAI Status:</strong> {openai_status}<br>
            <strong>📊 Model:</strong> GPT-3.5-turbo (when available)
        </div>
        
        <form id="codeForm">
            <div class="form-group">
                <label for="problem">Problem Description:</label>
                <input type="text" id="problem" name="problem_description" placeholder="Describe the coding problem..." required>
            </div>
            
            <div class="form-group">
                <label for="code">Python Code:</label>
                <textarea id="code" name="code" rows="15" placeholder="Enter your Python code here..." required></textarea>
            </div>
            
            <button type="submit" id="submitButton">🚀 Evaluate Code (AST + LLM)</button>
        </form>
        
        <div id="results" style="display: none;"></div>
    </div>
    
    <script>
        document.addEventListener('DOMContentLoaded', function() {{
            const form = document.getElementById('codeForm');
            const button = document.getElementById('submitButton');
            const results = document.getElementById('results');
            
            form.addEventListener('submit', function(event) {{
                event.preventDefault();
                event.stopPropagation();
                
                const formData = new FormData(form);
                const code = formData.get('code');
                const problem = formData.get('problem_description');
                
                if (!code || !problem) {{
                    alert('Please fill in both fields');
                    return;
                }}
                
                button.disabled = true;
                button.textContent = '⏳ Running AST + LLM Analysis...';
                results.style.display = 'block';
                results.innerHTML = '<div class="loading">🤖 Running comprehensive analysis with LLM...</div>';
                
                fetch('/evaluate', {{
                    method: 'POST',
                    body: formData
                }})
                .then(response => response.json())
                .then(data => {{
                    console.log('Analysis complete:', data);
                    displayResults(data);
                }})
                .catch(error => {{
                    console.error('Error:', error);
                    results.innerHTML = '<div class="error">❌ Error: ' + error.message + '</div>';
                }})
                .finally(() => {{
                    button.disabled = false;
                    button.textContent = '🚀 Evaluate Code (AST + LLM)';
                }});
            }});
            
            function displayResults(data) {{
                let html = '';
                
                if (data.overall_score !== undefined) {{
                    const score = data.overall_score;
                    let scoreClass = 'poor';
                    if (score >= 0.8) scoreClass = 'excellent';
                    else if (score >= 0.6) scoreClass = 'good';
                    else if (score >= 0.4) scoreClass = 'fair';
                    
                    html += '<div class="score ' + scoreClass + '">Combined Score: ' + (score * 100).toFixed(1) + '%</div>';
                    
                    // AST Analysis
                    if (data.ast_analysis) {{
                        html += '<div class="analysis-section">';
                        html += '<h4>🔧 AST Analysis</h4>';
                        html += '<div class="method">Method: ' + (data.ast_analysis.method || 'AST') + '</div>';
                        html += '<p>Score: ' + (data.ast_analysis.score * 100).toFixed(1) + '%</p>';
                        html += '<p>' + data.ast_analysis.feedback + '</p>';
                        if (data.ast_analysis.metrics) {{
                            html += '<p><small>Functions: ' + data.ast_analysis.metrics.functions + ', Classes: ' + data.ast_analysis.metrics.classes + ', Lines: ' + data.ast_analysis.metrics.lines + '</small></p>';
                        }}
                        html += '</div>';
                    }}
                    
                    // LLM Analysis
                    if (data.llm_analysis) {{
                        html += '<div class="analysis-section">';
                        html += '<h4>🤖 LLM Analysis</h4>';
                        html += '<div class="method">Model: ' + (data.llm_analysis.model || 'None') + '</div>';
                        html += '<p>Score: ' + (data.llm_analysis.score * 100).toFixed(1) + '%</p>';
                        html += '<p>' + data.llm_analysis.feedback + '</p>';
                        
                        if (data.llm_analysis.strengths && data.llm_analysis.strengths.length > 0) {{
                            html += '<h5>✅ Strengths:</h5><ul class="list">';
                            data.llm_analysis.strengths.forEach(strength => {{
                                html += '<li>' + strength + '</li>';
                            }});
                            html += '</ul>';
                        }}
                        
                        if (data.llm_analysis.improvements && data.llm_analysis.improvements.length > 0) {{
                            html += '<h5>⚠️ Improvements:</h5><ul class="list">';
                            data.llm_analysis.improvements.forEach(improvement => {{
                                html += '<li>' + improvement + '</li>';
                            }});
                            html += '</ul>';
                        }}
                        
                        html += '</div>';
                    }}
                    
                    html += '<div style="margin-top: 20px; padding: 15px; background: #e7f3ff; border-radius: 6px;">';
                    html += '<strong>Summary:</strong> ' + data.summary;
                    html += '</div>';
                }}
                
                results.innerHTML = html;
            }}
        }});
    </script>
</body>
</html>'''
    
    return HTMLResponse(content=html)

@app.get("/health")
async def health():
    return {
        "status": "healthy", 
        "version": "llm-enhanced",
        "openai_available": OPENAI_AVAILABLE,
        "openai_configured": evaluator.openai_client is not None
    }

@app.post("/evaluate")
async def evaluate_code(
    code: str = Form(...),
    problem_description: str = Form(...)
):
    logger.info(f"LLM evaluation request - Code: {len(code)} chars, Problem: {problem_description[:50]}...")
    
    try:
        results = await evaluator.comprehensive_evaluation(code, problem_description)
        logger.info(f"Evaluation successful: {results['overall_score']:.2f}")
        return JSONResponse(content=results)
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 3003))
    uvicorn.run(
        "llm_app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )
