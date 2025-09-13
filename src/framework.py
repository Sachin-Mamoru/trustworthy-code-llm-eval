"""
TrustworthyCodeLLM: Enhanced Multi-Modal Evaluation Framework

This module provides the core evaluation framework for assessing Code LLM trustworthiness
across multiple dimensions using hybrid evaluation methodologies.
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class TrustworthinessCategory(Enum):
    """Categories of trustworthiness evaluation."""
    COMMUNICATION = "communication"
    SECURITY = "security"
    ROBUSTNESS = "robustness"
    MAINTAINABILITY = "maintainability"
    ETHICAL = "ethical"


class EvaluationMethod(Enum):
    """Different evaluation methods available."""
    EXECUTION_BASED = "execution"
    STATIC_ANALYSIS = "static"
    LLM_ENSEMBLE = "llm_ensemble"
    HUMAN_VALIDATION = "human"


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    category: TrustworthinessCategory
    method: EvaluationMethod
    score: float
    confidence: float
    details: Dict[str, Any]
    timestamp: float
    
    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.time()


class CodeSample(BaseModel):
    """Represents a code sample for evaluation."""
    id: str
    source_code: str
    language: str = "python"
    problem_description: str = ""
    expected_output: Optional[str] = None
    test_cases: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TrustworthinessEvaluator(ABC):
    """Abstract base class for trustworthiness evaluators."""
    
    def __init__(self, category: TrustworthinessCategory):
        self.category = category
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
    
    @abstractmethod
    async def evaluate(self, code_sample: CodeSample) -> EvaluationResult:
        """Evaluate a code sample for trustworthiness."""
        pass
    
    @abstractmethod
    def get_evaluation_methods(self) -> List[EvaluationMethod]:
        """Return list of evaluation methods this evaluator supports."""
        pass


class MultiModalEvaluationFramework:
    """
    Core framework for multi-modal trustworthiness evaluation.
    
    This framework combines multiple evaluation approaches to provide
    reliable and comprehensive trustworthiness assessment.
    """
    
    def __init__(self):
        self.evaluators: Dict[TrustworthinessCategory, List[TrustworthinessEvaluator]] = {}
        self.evaluation_history: List[EvaluationResult] = []
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def register_evaluator(self, evaluator: TrustworthinessEvaluator):
        """Register an evaluator for a specific trustworthiness category."""
        if evaluator.category not in self.evaluators:
            self.evaluators[evaluator.category] = []
        self.evaluators[evaluator.category].append(evaluator)
        self.logger.info(f"Registered evaluator {evaluator.__class__.__name__} for {evaluator.category.value}")
    
    async def evaluate_code_sample(
        self, 
        code_sample: CodeSample,
        categories: Optional[List[TrustworthinessCategory]] = None
    ) -> Dict[TrustworthinessCategory, List[EvaluationResult]]:
        """
        Evaluate a code sample across specified trustworthiness categories.
        
        Args:
            code_sample: The code sample to evaluate
            categories: Categories to evaluate (all if None)
        
        Returns:
            Dictionary mapping categories to evaluation results
        """
        if categories is None:
            categories = list(self.evaluators.keys())
        
        results = {}
        
        for category in categories:
            if category not in self.evaluators:
                self.logger.warning(f"No evaluators registered for category: {category.value}")
                continue
            
            category_results = []
            
            # Run all evaluators for this category in parallel
            tasks = [
                evaluator.evaluate(code_sample) 
                for evaluator in self.evaluators[category]
            ]
            
            evaluation_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in evaluation_results:
                if isinstance(result, Exception):
                    self.logger.error(f"Evaluation failed: {result}")
                    continue
                
                category_results.append(result)
                self.evaluation_history.append(result)
            
            results[category] = category_results
        
        return results
    
    def compute_consensus_score(
        self, 
        results: List[EvaluationResult]
    ) -> Tuple[float, float]:
        """
        Compute consensus score and confidence from multiple evaluation results.
        
        Uses weighted averaging based on individual confidence scores.
        
        Args:
            results: List of evaluation results
        
        Returns:
            Tuple of (consensus_score, consensus_confidence)
        """
        if not results:
            return 0.0, 0.0
        
        # Weight scores by their confidence
        weighted_scores = []
        total_confidence = 0.0
        
        for result in results:
            weighted_scores.append(result.score * result.confidence)
            total_confidence += result.confidence
        
        if total_confidence == 0:
            # All evaluators have zero confidence
            consensus_score = np.mean([r.score for r in results])
            consensus_confidence = 0.0
        else:
            consensus_score = sum(weighted_scores) / total_confidence
            # Consensus confidence is the average of individual confidences
            consensus_confidence = total_confidence / len(results)
        
        return consensus_score, consensus_confidence
    
    def generate_trustworthiness_report(
        self, 
        evaluation_results: Dict[TrustworthinessCategory, List[EvaluationResult]]
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive trustworthiness report.
        
        Args:
            evaluation_results: Results from evaluate_code_sample
        
        Returns:
            Comprehensive report dictionary
        """
        report = {
            "timestamp": time.time(),
            "categories": {},
            "overall_score": 0.0,
            "overall_confidence": 0.0,
            "recommendations": []
        }
        
        category_scores = []
        category_confidences = []
        
        for category, results in evaluation_results.items():
            if not results:
                continue
            
            consensus_score, consensus_confidence = self.compute_consensus_score(results)
            
            # Identify potential issues based on low scores or high disagreement
            disagreement = np.std([r.score for r in results]) if len(results) > 1 else 0.0
            
            category_report = {
                "consensus_score": consensus_score,
                "consensus_confidence": consensus_confidence,
                "individual_results": [
                    {
                        "method": r.method.value,
                        "score": r.score,
                        "confidence": r.confidence,
                        "details": r.details
                    }
                    for r in results
                ],
                "disagreement": disagreement,
                "evaluation_count": len(results)
            }
            
            # Add category-specific recommendations
            if consensus_score < 0.5:
                report["recommendations"].append(
                    f"Low {category.value} trustworthiness detected. "
                    f"Score: {consensus_score:.2f}. Review needed."
                )
            
            if disagreement > 0.3:
                report["recommendations"].append(
                    f"High disagreement in {category.value} evaluation. "
                    f"Manual review recommended."
                )
            
            report["categories"][category.value] = category_report
            category_scores.append(consensus_score)
            category_confidences.append(consensus_confidence)
        
        # Calculate overall trustworthiness
        if category_scores:
            report["overall_score"] = np.mean(category_scores)
            report["overall_confidence"] = np.mean(category_confidences)
        
        return report
    
    def export_results(self, filepath: Path, format: str = "json"):
        """Export evaluation history to file."""
        if format == "json":
            with open(filepath, 'w') as f:
                json.dump([
                    {
                        "category": r.category.value,
                        "method": r.method.value,
                        "score": r.score,
                        "confidence": r.confidence,
                        "details": r.details,
                        "timestamp": r.timestamp
                    }
                    for r in self.evaluation_history
                ], f, indent=2)
        elif format == "csv":
            df = pd.DataFrame([
                {
                    "category": r.category.value,
                    "method": r.method.value,
                    "score": r.score,
                    "confidence": r.confidence,
                    "timestamp": r.timestamp
                }
                for r in self.evaluation_history
            ])
            df.to_csv(filepath, index=False)
        else:
            raise ValueError(f"Unsupported export format: {format}")


# Global framework instance
framework = MultiModalEvaluationFramework()
