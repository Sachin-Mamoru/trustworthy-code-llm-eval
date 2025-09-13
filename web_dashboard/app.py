"""
TrustworthyCodeLLM Web Dashboard

Interactive web interface for real-time code LLM trustworthiness evaluation.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import plotly.graph_objects as go
import plotly.express as px
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from pydantic import BaseModel

# Import our evaluation framework
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.framework import MultiModalEvaluationFramework, CodeSample, TrustworthinessCategory
from src.evaluators.enhanced_communication import EnhancedCommunicationEvaluator
from src.evaluators.execution_based import (
    ExecutionBasedRobustnessEvaluator,
    ExecutionBasedSecurityEvaluator,
    ExecutionBasedPerformanceEvaluator
)
from src.evaluators.static_analysis import (
    StaticSecurityAnalyzer,
    StaticMaintainabilityAnalyzer,
    StaticEthicalAnalyzer
)


# Initialize FastAPI app
app = FastAPI(
    title="TrustworthyCodeLLM Dashboard",
    description="Interactive evaluation platform for Code LLM trustworthiness assessment",
    version="1.0.0"
)

# Setup templates and static files
templates = Jinja2Templates(directory="web_dashboard/templates")
app.mount("/static", StaticFiles(directory="web_dashboard/static"), name="static")

# Initialize evaluation framework
framework = MultiModalEvaluationFramework()
framework.register_evaluator(EnhancedCommunicationEvaluator())
framework.register_evaluator(ExecutionBasedRobustnessEvaluator())
framework.register_evaluator(ExecutionBasedSecurityEvaluator())
framework.register_evaluator(ExecutionBasedPerformanceEvaluator())
framework.register_evaluator(StaticSecurityAnalyzer())
framework.register_evaluator(StaticMaintainabilityAnalyzer())
framework.register_evaluator(StaticEthicalAnalyzer())

# Global state for real-time updates
evaluation_results = []
active_connections: List[WebSocket] = []


class EvaluationRequest(BaseModel):
    """Request model for code evaluation."""
    code: str
    problem_description: str = ""
    language: str = "python"


class EvaluationResponse(BaseModel):
    """Response model for evaluation results."""
    id: str
    timestamp: datetime
    overall_score: float
    category_scores: Dict[str, float]
    confidence_scores: Dict[str, float]
    details: Dict[str, any]


@app.get("/", response_class=HTMLResponse)
async def dashboard_home(request: Request):
    """Main dashboard page."""
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "title": "TrustworthyCodeLLM Dashboard"
    })


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "framework": "operational"}


@app.post("/api/evaluate", response_model=EvaluationResponse)
async def evaluate_code(request: EvaluationRequest):
    """Evaluate code sample for trustworthiness."""
    try:
        # Create code sample
        sample = CodeSample(
            id=f"web_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            source_code=request.code,
            problem_description=request.problem_description,
            language=request.language
        )
        
        # Run evaluation
        results = await framework.evaluate_code_sample(sample)
        
        # Process results
        category_scores = {}
        confidence_scores = {}
        all_details = {}
        
        overall_score = 0.0
        total_weight = 0.0
        
        for result in results:
            category = result.category.value
            category_scores[category] = result.score
            confidence_scores[category] = result.confidence
            all_details[category] = result.details
            
            # Weighted average for overall score
            weight = result.confidence
            overall_score += result.score * weight
            total_weight += weight
        
        if total_weight > 0:
            overall_score /= total_weight
        
        response = EvaluationResponse(
            id=sample.id,
            timestamp=datetime.now(),
            overall_score=overall_score,
            category_scores=category_scores,
            confidence_scores=confidence_scores,
            details=all_details
        )
        
        # Store results globally
        evaluation_results.append(response.dict())
        
        # Broadcast to active WebSocket connections
        await broadcast_update(response.dict())
        
        return response
        
    except Exception as e:
        logging.error(f"Evaluation error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": "Evaluation failed", "message": str(e)}
        )


@app.get("/api/results")
async def get_recent_results(limit: int = 50):
    """Get recent evaluation results."""
    return evaluation_results[-limit:]


@app.get("/api/analytics")
async def get_analytics():
    """Get analytics and trends from evaluation history."""
    if not evaluation_results:
        return {"message": "No data available"}
    
    # Calculate trends
    recent_results = evaluation_results[-100:]  # Last 100 evaluations
    
    # Overall score trend
    scores = [r["overall_score"] for r in recent_results]
    timestamps = [r["timestamp"] for r in recent_results]
    
    # Category analysis
    categories = list(TrustworthinessCategory)
    category_averages = {}
    
    for category in categories:
        cat_name = category.value
        cat_scores = [
            r["category_scores"].get(cat_name, 0) 
            for r in recent_results 
            if cat_name in r["category_scores"]
        ]
        if cat_scores:
            category_averages[cat_name] = sum(cat_scores) / len(cat_scores)
    
    return {
        "total_evaluations": len(evaluation_results),
        "average_overall_score": sum(scores) / len(scores) if scores else 0,
        "category_averages": category_averages,
        "recent_trend": "improving" if len(scores) >= 2 and scores[-1] > scores[0] else "stable",
        "score_distribution": {
            "excellent": len([s for s in scores if s >= 0.8]),
            "good": len([s for s in scores if 0.6 <= s < 0.8]),
            "fair": len([s for s in scores if 0.4 <= s < 0.6]),
            "poor": len([s for s in scores if s < 0.4])
        }
    }


@app.get("/api/visualization/score_trend")
async def score_trend_chart():
    """Generate score trend visualization."""
    if not evaluation_results:
        return {"error": "No data available"}
    
    recent_results = evaluation_results[-50:]
    
    fig = go.Figure()
    
    # Overall score trend
    fig.add_trace(go.Scatter(
        x=[r["timestamp"] for r in recent_results],
        y=[r["overall_score"] for r in recent_results],
        mode='lines+markers',
        name='Overall Score',
        line=dict(color='#1f77b4', width=3)
    ))
    
    # Category trends
    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for i, category in enumerate(TrustworthinessCategory):
        cat_name = category.value
        y_values = [
            r["category_scores"].get(cat_name, None) 
            for r in recent_results
        ]
        
        # Filter out None values
        x_filtered = []
        y_filtered = []
        for j, y in enumerate(y_values):
            if y is not None:
                x_filtered.append(recent_results[j]["timestamp"])
                y_filtered.append(y)
        
        if y_filtered:
            fig.add_trace(go.Scatter(
                x=x_filtered,
                y=y_filtered,
                mode='lines',
                name=cat_name.title(),
                line=dict(color=colors[i % len(colors)], width=2),
                opacity=0.7
            ))
    
    fig.update_layout(
        title="Trustworthiness Score Trends",
        xaxis_title="Time",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1]),
        template="plotly_white",
        height=400
    )
    
    return fig.to_json()


@app.get("/api/visualization/category_radar")
async def category_radar_chart():
    """Generate category comparison radar chart."""
    if not evaluation_results:
        return {"error": "No data available"}
    
    # Calculate averages for each category
    recent_results = evaluation_results[-20:]
    categories = list(TrustworthinessCategory)
    
    averages = []
    category_names = []
    
    for category in categories:
        cat_name = category.value
        scores = [
            r["category_scores"].get(cat_name, 0) 
            for r in recent_results 
            if cat_name in r["category_scores"]
        ]
        if scores:
            averages.append(sum(scores) / len(scores))
            category_names.append(cat_name.title())
    
    # Close the radar chart
    averages.append(averages[0])
    category_names.append(category_names[0])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=averages,
        theta=category_names,
        fill='toself',
        name='Average Scores',
        line_color='#1f77b4'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        title="Trustworthiness Categories Overview",
        template="plotly_white"
    )
    
    return fig.to_json()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        active_connections.remove(websocket)


async def broadcast_update(data: dict):
    """Broadcast updates to all connected WebSocket clients."""
    if active_connections:
        message = json.dumps(data)
        for connection in active_connections.copy():
            try:
                await connection.send_text(message)
            except Exception:
                active_connections.remove(connection)


@app.get("/leaderboard")
async def leaderboard(request: Request):
    """Leaderboard page for model comparison."""
    return templates.TemplateResponse("leaderboard.html", {
        "request": request,
        "title": "TrustworthyCodeLLM Leaderboard"
    })


@app.get("/documentation")
async def documentation(request: Request):
    """Documentation page."""
    return templates.TemplateResponse("documentation.html", {
        "request": request,
        "title": "Documentation"
    })


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
