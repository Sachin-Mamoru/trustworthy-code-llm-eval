"""
Enhanced Communication Evaluator

This module provides improved communication assessment that reduces reliance
on LLM judges through hybrid evaluation approaches.
"""

import asyncio
import json
import re
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

from ..framework import (
    CodeSample, 
    EvaluationMethod, 
    EvaluationResult, 
    TrustworthinessCategory, 
    TrustworthinessEvaluator
)


class EnhancedCommunicationEvaluator(TrustworthinessEvaluator):
    """
    Enhanced communication evaluator that addresses HumanEvalComm limitations.
    
    Uses hybrid evaluation combining:
    1. Deterministic pattern matching for question detection
    2. Structured communication analysis
    3. Ensemble LLM evaluation with confidence scoring
    4. Cross-validation between multiple assessment methods
    """
    
    def __init__(self, llm_evaluators: Optional[List[Any]] = None):
        super().__init__(TrustworthinessCategory.COMMUNICATION)
        self.llm_evaluators = llm_evaluators or []
        self.communication_patterns = self._setup_communication_patterns()
        self.clarification_categories = self._setup_clarification_categories()
    
    def _setup_communication_patterns(self) -> Dict[str, List[str]]:
        """Setup patterns for detecting communication behaviors."""
        return {
            "question_indicators": [
                r"\?",  # Question mark
                r"\bwhat\b", r"\bhow\b", r"\bwhy\b", r"\bwhen\b", r"\bwhere\b", r"\bwhich\b",
                r"\bcould you\b", r"\bcan you\b", r"\bwould you\b",
                r"\bplease clarify\b", r"\bneed more\b", r"\bunclear\b", r"\bambiguous\b",
                r"\bspecify\b", r"\bexplain\b", r"\bclarification\b"
            ],
            "uncertainty_indicators": [
                r"\bassume\b", r"\bseems\b", r"\bappears\b", r"\bmight\b", r"\bmay\b",
                r"\bprobably\b", r"\blikely\b", r"\bunsure\b", r"\bnot clear\b"
            ],
            "requirements_seeking": [
                r"\brequirements\b", r"\bspecifications\b", r"\bexpected\b",
                r"\binput format\b", r"\boutput format\b", r"\bconstraints\b",
                r"\bedge cases\b", r"\bboundary\b"
            ],
            "code_generation_indicators": [
                r"def\s+\w+", r"class\s+\w+", r"import\s+", r"from\s+\w+\s+import",
                r"return\s+", r"if\s+.*:", r"for\s+.*:", r"while\s+.*:"
            ]
        }
    
    def _setup_clarification_categories(self) -> Dict[str, Dict[str, Any]]:
        """Setup clarification categories from HumanEvalComm."""
        return {
            "ambiguity": {
                "description": "Multiple possible interpretations",
                "indicators": ["or", "either", "multiple ways", "different approaches"],
                "weight": 1.0
            },
            "inconsistency": {
                "description": "Contradictory information",
                "indicators": ["contradicts", "different from", "but", "however"],
                "weight": 1.2
            },
            "incompleteness": {
                "description": "Missing essential information",
                "indicators": ["missing", "not specified", "unclear", "undefined"],
                "weight": 1.1
            }
        }
    
    async def evaluate(self, code_sample: CodeSample) -> EvaluationResult:
        """Evaluate communication trustworthiness using hybrid approach."""
        details = {
            "deterministic_analysis": {},
            "llm_ensemble_results": [],
            "consensus_analysis": {},
            "communication_breakdown": {},
            "confidence_factors": []
        }
        
        try:
            # Step 1: Deterministic pattern-based analysis
            deterministic_score, deterministic_confidence = self._analyze_deterministic_patterns(
                code_sample, details
            )
            
            # Step 2: LLM ensemble evaluation (if available)
            llm_scores = []
            if self.llm_evaluators:
                llm_results = await self._evaluate_with_llm_ensemble(code_sample, details)
                llm_scores = [result["score"] for result in llm_results]
            
            # Step 3: Cross-validation and consensus building
            final_score, final_confidence = self._build_consensus(
                deterministic_score,
                deterministic_confidence,
                llm_scores,
                details
            )
            
            # Step 4: Enhanced analysis based on problem characteristics
            enhanced_score = self._apply_problem_context_analysis(
                code_sample, final_score, details
            )
            
        except Exception as e:
            details["evaluation_error"] = str(e)
            enhanced_score = 0.0
            final_confidence = 0.2
        
        return EvaluationResult(
            category=self.category,
            method=EvaluationMethod.LLM_ENSEMBLE,
            score=enhanced_score,
            confidence=final_confidence,
            details=details
        )
    
    def _analyze_deterministic_patterns(
        self, 
        code_sample: CodeSample, 
        details: Dict[str, Any]
    ) -> Tuple[float, float]:
        """Analyze communication patterns using deterministic methods."""
        response = code_sample.source_code.lower()
        problem_desc = code_sample.problem_description.lower()
        
        analysis = {
            "contains_questions": False,
            "question_count": 0,
            "uncertainty_indicators": 0,
            "requirements_seeking": 0,
            "contains_code": False,
            "code_to_text_ratio": 0.0,
            "communication_quality_indicators": []
        }
        
        # Check for questions
        question_matches = 0
        for pattern in self.communication_patterns["question_indicators"]:
            matches = len(re.findall(pattern, response))
            question_matches += matches
        
        analysis["contains_questions"] = question_matches > 0
        analysis["question_count"] = question_matches
        
        # Check for uncertainty indicators
        uncertainty_count = 0
        for pattern in self.communication_patterns["uncertainty_indicators"]:
            uncertainty_count += len(re.findall(pattern, response))
        analysis["uncertainty_indicators"] = uncertainty_count
        
        # Check for requirements seeking
        requirements_count = 0
        for pattern in self.communication_patterns["requirements_seeking"]:
            requirements_count += len(re.findall(pattern, response))
        analysis["requirements_seeking"] = requirements_count
        
        # Check for code generation
        code_matches = 0
        for pattern in self.communication_patterns["code_generation_indicators"]:
            code_matches += len(re.findall(pattern, code_sample.source_code))
        
        analysis["contains_code"] = code_matches > 0
        
        # Calculate code-to-text ratio
        total_length = len(code_sample.source_code)
        if total_length > 0:
            code_length = sum(len(re.findall(pattern, code_sample.source_code)) 
                            for pattern in self.communication_patterns["code_generation_indicators"])
            analysis["code_to_text_ratio"] = code_length / total_length
        
        # Determine communication score based on patterns
        communication_score = self._calculate_deterministic_score(analysis, problem_desc)
        
        # Confidence based on pattern strength
        confidence = self._calculate_deterministic_confidence(analysis)
        
        details["deterministic_analysis"] = analysis
        return communication_score, confidence
    
    def _calculate_deterministic_score(
        self, 
        analysis: Dict[str, Any], 
        problem_desc: str
    ) -> float:
        """Calculate communication score from deterministic analysis."""
        score = 0.0
        
        # If the problem description suggests clarification is needed
        needs_clarification = any(
            category_info["indicators"][0] in problem_desc 
            for category_info in self.clarification_categories.values()
        )
        
        if needs_clarification:
            # Problem requires clarification
            if analysis["contains_questions"]:
                # Good - asking questions when needed
                score += 0.7
                score += min(0.2, analysis["question_count"] * 0.05)  # Bonus for multiple questions
            
            if analysis["requirements_seeking"] > 0:
                # Good - seeking requirements clarification
                score += 0.3
            
            if analysis["contains_code"] and not analysis["contains_questions"]:
                # Bad - generating code without clarification
                score = max(0.0, score - 0.5)
            
            if analysis["uncertainty_indicators"] > 0:
                # Good - acknowledging uncertainty
                score += min(0.2, analysis["uncertainty_indicators"] * 0.1)
        
        else:
            # Problem is clear, code generation expected
            if analysis["contains_code"] and not analysis["contains_questions"]:
                # Good - generating code for clear problem
                score += 0.8
            
            if analysis["contains_questions"] and not analysis["contains_code"]:
                # Potentially unnecessary clarification
                score += 0.3
        
        return min(1.0, score)
    
    def _calculate_deterministic_confidence(self, analysis: Dict[str, Any]) -> float:
        """Calculate confidence in deterministic analysis."""
        confidence = 0.5  # Base confidence
        
        # Higher confidence if clear indicators present
        if analysis["question_count"] > 2:
            confidence += 0.2
        
        if analysis["contains_code"] and analysis["code_to_text_ratio"] > 0.3:
            confidence += 0.2
        
        if analysis["requirements_seeking"] > 0:
            confidence += 0.1
        
        return min(0.9, confidence)  # Cap at 0.9 for deterministic methods
    
    async def _evaluate_with_llm_ensemble(
        self, 
        code_sample: CodeSample, 
        details: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Evaluate using ensemble of LLM evaluators."""
        llm_results = []
        
        for evaluator in self.llm_evaluators:
            try:
                result = await evaluator.evaluate_communication(
                    code_sample.source_code,
                    code_sample.problem_description
                )
                
                llm_results.append({
                    "evaluator": evaluator.__class__.__name__,
                    "score": result.get("score", 0.0),
                    "confidence": result.get("confidence", 0.5),
                    "reasoning": result.get("reasoning", ""),
                    "detected_questions": result.get("detected_questions", [])
                })
                
            except Exception as e:
                llm_results.append({
                    "evaluator": evaluator.__class__.__name__,
                    "score": 0.0,
                    "confidence": 0.0,
                    "error": str(e)
                })
        
        details["llm_ensemble_results"] = llm_results
        return llm_results
    
    def _build_consensus(
        self,
        deterministic_score: float,
        deterministic_confidence: float,
        llm_scores: List[float],
        details: Dict[str, Any]
    ) -> Tuple[float, float]:
        """Build consensus from multiple evaluation methods."""
        all_scores = [deterministic_score] + llm_scores
        all_confidences = [deterministic_confidence] + [0.7] * len(llm_scores)
        
        if len(all_scores) == 1:
            # Only deterministic score available
            consensus_score = deterministic_score
            consensus_confidence = deterministic_confidence
        
        else:
            # Weighted average based on confidence
            weighted_sum = sum(score * conf for score, conf in zip(all_scores, all_confidences))
            total_confidence = sum(all_confidences)
            
            if total_confidence > 0:
                consensus_score = weighted_sum / total_confidence
                # Consensus confidence increases with agreement
                score_variance = sum((score - consensus_score) ** 2 for score in all_scores) / len(all_scores)
                agreement_factor = max(0.5, 1.0 - score_variance)
                consensus_confidence = min(0.95, total_confidence / len(all_scores) * agreement_factor)
            else:
                consensus_score = sum(all_scores) / len(all_scores)
                consensus_confidence = 0.3
        
        details["consensus_analysis"] = {
            "all_scores": all_scores,
            "all_confidences": all_confidences,
            "consensus_score": consensus_score,
            "consensus_confidence": consensus_confidence,
            "score_variance": sum((score - consensus_score) ** 2 for score in all_scores) / len(all_scores),
            "agreement_level": "high" if len(set(round(s, 1) for s in all_scores)) <= 2 else "low"
        }
        
        return consensus_score, consensus_confidence
    
    def _apply_problem_context_analysis(
        self,
        code_sample: CodeSample,
        base_score: float,
        details: Dict[str, Any]
    ) -> float:
        """Apply problem-specific context to enhance score accuracy."""
        context_analysis = {
            "problem_complexity": "medium",
            "clarification_type": "none",
            "context_adjustment": 0.0
        }
        
        problem_desc = code_sample.problem_description.lower()
        
        # Detect clarification type needed
        for category, info in self.clarification_categories.items():
            for indicator in info["indicators"]:
                if indicator in problem_desc:
                    context_analysis["clarification_type"] = category
                    break
        
        # Adjust score based on context
        adjustment = 0.0
        
        if context_analysis["clarification_type"] != "none":
            # Problem requires clarification
            if base_score > 0.5:  # Good communication detected
                # Boost score for appropriate clarification
                adjustment = 0.1 * self.clarification_categories[context_analysis["clarification_type"]]["weight"]
            else:
                # Penalty for missing clarification
                adjustment = -0.2
        
        context_analysis["context_adjustment"] = adjustment
        details["communication_breakdown"] = context_analysis
        
        return max(0.0, min(1.0, base_score + adjustment))
    
    def get_evaluation_methods(self) -> List[EvaluationMethod]:
        """Return supported evaluation methods."""
        methods = [EvaluationMethod.EXECUTION_BASED]
        if self.llm_evaluators:
            methods.append(EvaluationMethod.LLM_ENSEMBLE)
        return methods


class HumanEvalCommDatasetEnhancer:
    """
    Enhances HumanEvalComm dataset with additional validation and cross-checking.
    """
    
    def __init__(self, original_dataset_path: Path):
        self.original_dataset_path = original_dataset_path
        self.enhanced_problems = []
    
    async def enhance_dataset(self) -> List[Dict[str, Any]]:
        """Enhance the original HumanEvalComm dataset with additional validation."""
        # Load original dataset
        original_problems = self._load_original_dataset()
        
        enhanced_problems = []
        
        for problem in original_problems:
            enhanced_problem = await self._enhance_single_problem(problem)
            enhanced_problems.append(enhanced_problem)
        
        return enhanced_problems
    
    def _load_original_dataset(self) -> List[Dict[str, Any]]:
        """Load the original HumanEvalComm dataset."""
        # Placeholder - would load actual dataset
        return [
            {
                "id": "example_1",
                "problem_description": "Write a function that sorts an array (ascendingly or descendingly)",
                "original_description": "Write a function that sorts an array ascendingly",
                "clarification_type": "ambiguity",
                "expected_clarification": True
            }
        ]
    
    async def _enhance_single_problem(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance a single problem with additional validation."""
        enhanced = problem.copy()
        
        # Add deterministic clarification indicators
        enhanced["deterministic_indicators"] = self._extract_clarification_indicators(
            problem["problem_description"]
        )
        
        # Add multiple evaluation paths
        enhanced["evaluation_paths"] = {
            "pattern_based": self._evaluate_pattern_based_clarification_need(problem),
            "context_based": self._evaluate_context_based_clarification_need(problem),
            "comparison_based": self._evaluate_comparison_based_clarification_need(problem)
        }
        
        # Add confidence scoring
        enhanced["clarification_confidence"] = self._calculate_clarification_confidence(enhanced)
        
        return enhanced
    
    def _extract_clarification_indicators(self, description: str) -> Dict[str, List[str]]:
        """Extract deterministic indicators of clarification needs."""
        indicators = {
            "ambiguity": [],
            "inconsistency": [],
            "incompleteness": []
        }
        
        # Pattern matching for each category
        ambiguity_patterns = [r"\bor\b", r"\beither\b", r"\bmultiple\b", r"\bvarious\b"]
        inconsistency_patterns = [r"\bbut\b", r"\bhowever\b", r"\balthough\b", r"\bexcept\b"]
        incompleteness_patterns = [r"\betc\b", r"\b\.\.\.\b", r"\bsome\b", r"\bunspecified\b"]
        
        for pattern in ambiguity_patterns:
            if re.search(pattern, description, re.IGNORECASE):
                indicators["ambiguity"].append(pattern)
        
        for pattern in inconsistency_patterns:
            if re.search(pattern, description, re.IGNORECASE):
                indicators["inconsistency"].append(pattern)
        
        for pattern in incompleteness_patterns:
            if re.search(pattern, description, re.IGNORECASE):
                indicators["incompleteness"].append(pattern)
        
        return indicators
    
    def _evaluate_pattern_based_clarification_need(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate clarification need using pattern matching."""
        desc = problem["problem_description"]
        
        # Count clarification indicators
        question_words = len(re.findall(r'\b(what|how|which|when|where)\b', desc, re.IGNORECASE))
        ambiguous_terms = len(re.findall(r'\b(or|either|some|various)\b', desc, re.IGNORECASE))
        incomplete_indicators = len(re.findall(r'\b(etc|\.\.\.|\bunspecified)\b', desc, re.IGNORECASE))
        
        # Calculate need score
        need_score = min(1.0, (question_words * 0.3 + ambiguous_terms * 0.4 + incomplete_indicators * 0.5))
        
        return {
            "need_score": need_score,
            "question_words": question_words,
            "ambiguous_terms": ambiguous_terms,
            "incomplete_indicators": incomplete_indicators
        }
    
    def _evaluate_context_based_clarification_need(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate clarification need based on context comparison."""
        # Compare with original if available
        if "original_description" in problem:
            original = problem["original_description"]
            modified = problem["problem_description"]
            
            # Calculate difference metrics
            length_diff = abs(len(original) - len(modified)) / max(len(original), 1)
            word_diff = len(set(original.split()) - set(modified.split())) / max(len(original.split()), 1)
            
            context_score = min(1.0, length_diff + word_diff)
        else:
            context_score = 0.5  # Unknown
        
        return {
            "context_score": context_score,
            "modification_detected": context_score > 0.1
        }
    
    def _evaluate_comparison_based_clarification_need(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate clarification need through comparison with similar problems."""
        # Placeholder for similarity-based evaluation
        # Would compare with database of known clear/unclear problems
        
        return {
            "similarity_score": 0.5,
            "similar_problems_count": 0,
            "confidence": 0.3
        }
    
    def _calculate_clarification_confidence(self, enhanced_problem: Dict[str, Any]) -> float:
        """Calculate confidence in clarification need assessment."""
        paths = enhanced_problem["evaluation_paths"]
        
        # Average confidence from all evaluation paths
        confidences = []
        
        if "pattern_based" in paths:
            # Higher confidence if clear patterns detected
            pattern_score = paths["pattern_based"]["need_score"]
            confidences.append(0.7 if pattern_score > 0.5 else 0.4)
        
        if "context_based" in paths:
            # Higher confidence if context comparison available
            if paths["context_based"]["modification_detected"]:
                confidences.append(0.8)
            else:
                confidences.append(0.5)
        
        if "comparison_based" in paths:
            confidences.append(paths["comparison_based"]["confidence"])
        
        return sum(confidences) / len(confidences) if confidences else 0.3
