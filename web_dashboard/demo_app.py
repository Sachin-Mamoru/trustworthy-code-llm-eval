"""
Simplified TrustworthyCodeLLM Web Dashboard for Live Demo

This is a working demonstration version that showcases the UI and basic functionality.
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import json
import time
import asyncio
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import random

# Initialize FastAPI app
app = FastAPI(
    title="TrustworthyCodeLLM Dashboard",
    description="Multi-Modal Code LLM Trustworthiness Evaluation Framework",
    version="1.0.0"
)

# Setup templates
templates = Jinja2Templates(directory="web_dashboard/templates")

# In-memory storage for demo
evaluation_history = []

# Mock evaluation function for demo
async def mock_evaluate_code(code: str, problem_description: str, language: str) -> Dict[str, Any]:
    """Mock evaluation function that simulates the real evaluation process."""
    
    # Simulate processing time
    await asyncio.sleep(2)
    
    # Mock scores based on code analysis
    base_score = 0.7 + random.uniform(-0.2, 0.3)
    
    # Simple heuristics for demo
    if "def " in code and "return" in code:
        base_score += 0.1
    if "try:" in code or "except:" in code:
        base_score += 0.1
    if len(code.split('\n')) > 10:
        base_score += 0.05
    if "import" in code:
        base_score += 0.05
    
    # Clamp score between 0 and 1
    base_score = max(0.0, min(1.0, base_score))
    
    # Generate category scores with some variance
    categories = {
        "security": base_score + random.uniform(-0.1, 0.1),
        "robustness": base_score + random.uniform(-0.1, 0.1),
        "maintainability": base_score + random.uniform(-0.1, 0.1),
        "performance": base_score + random.uniform(-0.1, 0.1),
        "communication": base_score + random.uniform(-0.1, 0.1),
        "ethical": base_score + random.uniform(-0.1, 0.1)
    }
    
    # Clamp all scores
    for key in categories:
        categories[key] = max(0.0, min(1.0, categories[key]))
    
    # Calculate confidence scores
    confidence_scores = {
        category: random.uniform(0.6, 0.95) for category in categories
    }
    
    # Calculate overall score
    overall_score = sum(categories.values()) / len(categories)
    
    result = {
        "id": f"eval_{int(time.time())}_{random.randint(1000, 9999)}",
        "timestamp": datetime.now().isoformat(),
        "overall_score": overall_score,
        "category_scores": categories,
        "confidence_scores": confidence_scores,
        "code": code[:100] + "..." if len(code) > 100 else code,
        "language": language,
        "problem_description": problem_description
    }
    
    return result


@app.get("/", response_class=HTMLResponse)
async def dashboard_home(request: Request):
    """Main dashboard page."""
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "title": "Dashboard"
    })


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/api/evaluate")
async def evaluate_code(request: Request):
    """Evaluate code trustworthiness."""
    try:
        data = await request.json()
        code = data.get("code", "")
        problem_description = data.get("problem_description", "")
        language = data.get("language", "python")
        
        if not code.strip():
            raise HTTPException(status_code=400, detail="Code cannot be empty")
        
        # Run evaluation
        result = await mock_evaluate_code(code, problem_description, language)
        
        # Store in history
        evaluation_history.append(result)
        
        # Keep only last 50 evaluations
        if len(evaluation_history) > 50:
            evaluation_history.pop(0)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/visualization/score_trend")
async def get_score_trend():
    """Get score trend visualization data."""
    try:
        if len(evaluation_history) < 2:
            return {"error": "Not enough data for trend analysis"}
        
        # Get last 20 evaluations
        recent_evals = evaluation_history[-20:]
        
        timestamps = [eval_data["timestamp"] for eval_data in recent_evals]
        scores = [eval_data["overall_score"] * 100 for eval_data in recent_evals]
        
        chart_data = {
            "data": [{
                "x": timestamps,
                "y": scores,
                "type": "scatter",
                "mode": "lines+markers",
                "name": "Trustworthiness Score",
                "line": {"color": "#3498db", "width": 3},
                "marker": {"size": 8, "color": "#2c3e50"}
            }],
            "layout": {
                "title": "Score Trend Over Time",
                "xaxis": {"title": "Time"},
                "yaxis": {"title": "Score (%)", "range": [0, 100]},
                "template": "plotly_white"
            }
        }
        
        return json.dumps(chart_data)
        
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/visualization/category_radar")
async def get_category_radar():
    """Get category radar chart data."""
    try:
        if not evaluation_history:
            return {"error": "No evaluation data available"}
        
        # Get latest evaluation
        latest_eval = evaluation_history[-1]
        categories = list(latest_eval["category_scores"].keys())
        scores = [latest_eval["category_scores"][cat] * 100 for cat in categories]
        
        # Add first category at the end to close the radar
        categories.append(categories[0])
        scores.append(scores[0])
        
        chart_data = {
            "data": [{
                "type": "scatterpolar",
                "r": scores,
                "theta": categories,
                "fill": "toself",
                "name": "Latest Evaluation",
                "line": {"color": "#27ae60"},
                "fillcolor": "rgba(39, 174, 96, 0.3)"
            }],
            "layout": {
                "polar": {
                    "radialaxis": {
                        "visible": True,
                        "range": [0, 100]
                    }
                },
                "title": "Category Breakdown",
                "template": "plotly_white"
            }
        }
        
        return json.dumps(chart_data)
        
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/history")
async def get_evaluation_history():
    """Get evaluation history."""
    return {"evaluations": evaluation_history[-10:]}  # Last 10 evaluations


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
