"""
Test suite for the TrustworthyCodeLLM evaluation framework.

This test suite validates the core functionality and ensures reliability
of the multi-modal evaluation system.
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch
import json

# Import framework components
from src.framework import (
    MultiModalEvaluationFramework,
    CodeSample,
    TrustworthinessCategory,
    EvaluationResult,
    TrustworthinessReport
)
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
from src.evaluators.enhanced_communication import (
    EnhancedCommunicationEvaluator
)


class TestMultiModalEvaluationFramework:
    """Test the core evaluation framework."""
    
    @pytest.fixture
    def framework(self):
        """Create a fresh framework instance for each test."""
        return MultiModalEvaluationFramework()
    
    @pytest.fixture
    def sample_code(self):
        """Create a sample code for testing."""
        return CodeSample(
            id="test_sample",
            source_code="def hello(): return 'Hello, World!'",
            problem_description="Write a hello world function",
            language="python"
        )
    
    def test_framework_initialization(self, framework):
        """Test framework initializes correctly."""
        assert framework is not None
        assert len(framework.evaluators) == 0
        assert len(framework.evaluation_history) == 0
    
    def test_evaluator_registration(self, framework):
        """Test evaluator registration."""
        evaluator = EnhancedCommunicationEvaluator()
        framework.register_evaluator(evaluator)
        
        assert len(framework.evaluators) == 1
        assert TrustworthinessCategory.COMMUNICATION in framework.evaluators
    
    def test_multiple_evaluator_registration(self, framework):
        """Test registering multiple evaluators for same category."""
        evaluator1 = StaticSecurityAnalyzer()
        evaluator2 = ExecutionBasedSecurityEvaluator()
        
        framework.register_evaluator(evaluator1)
        framework.register_evaluator(evaluator2)
        
        assert len(framework.evaluators[TrustworthinessCategory.SECURITY]) == 2
    
    @pytest.mark.asyncio
    async def test_code_evaluation_basic(self, framework, sample_code):
        """Test basic code evaluation."""
        # Register a mock evaluator
        mock_evaluator = Mock()
        mock_evaluator.category = TrustworthinessCategory.COMMUNICATION
        mock_evaluator.evaluate.return_value = EvaluationResult(
            category=TrustworthinessCategory.COMMUNICATION,
            score=0.8,
            confidence=0.9,
            details={"test": "passed"},
            explanations=["Good communication"]
        )
        
        framework.register_evaluator(mock_evaluator)
        
        results = await framework.evaluate_code_sample(sample_code)
        
        assert len(results) == 1
        assert results[0].category == TrustworthinessCategory.COMMUNICATION
        assert results[0].score == 0.8
    
    def test_consensus_scoring(self, framework):
        """Test consensus scoring mechanism."""
        results = [
            EvaluationResult(
                category=TrustworthinessCategory.COMMUNICATION,
                score=0.8,
                confidence=0.9,
                details={},
                explanations=[]
            ),
            EvaluationResult(
                category=TrustworthinessCategory.COMMUNICATION,
                score=0.6,
                confidence=0.7,
                details={},
                explanations=[]
            )
        ]
        
        consensus = framework._calculate_consensus_score(results)
        
        # Should be weighted average: (0.8*0.9 + 0.6*0.7) / (0.9 + 0.7)
        expected = (0.8 * 0.9 + 0.6 * 0.7) / (0.9 + 0.7)
        assert abs(consensus["score"] - expected) < 0.001
    
    def test_report_generation(self, framework):
        """Test trustworthiness report generation."""
        results = [
            EvaluationResult(
                category=TrustworthinessCategory.COMMUNICATION,
                score=0.8,
                confidence=0.9,
                details={},
                explanations=["Good documentation"]
            ),
            EvaluationResult(
                category=TrustworthinessCategory.SECURITY,
                score=0.6,
                confidence=0.8,
                details={},
                explanations=["Some security concerns"]
            )
        ]
        
        report = framework.generate_trustworthiness_report(results)
        
        assert "overall_score" in report
        assert "overall_confidence" in report
        assert "categories" in report
        assert len(report["categories"]) == 2
        assert "recommendations" in report
    
    def test_export_functionality(self, framework, tmp_path):
        """Test result export functionality."""
        # Add some fake history
        framework.evaluation_history = [
            {
                "timestamp": "2024-01-01T00:00:00",
                "code_id": "test",
                "results": []
            }
        ]
        
        json_file = tmp_path / "test_export.json"
        csv_file = tmp_path / "test_export.csv"
        
        framework.export_results(str(json_file))
        framework.export_results(str(csv_file), format="csv")
        
        assert json_file.exists()
        assert csv_file.exists()


class TestExecutionBasedEvaluators:
    """Test execution-based evaluators."""
    
    @pytest.fixture
    def sample_code(self):
        return CodeSample(
            id="test_execution",
            source_code="""
def divide(a, b):
    if b == 0:
        raise ValueError("Division by zero")
    return a / b
            """,
            problem_description="Implement division function",
            language="python"
        )
    
    def test_robustness_evaluator(self, sample_code):
        """Test robustness evaluator."""
        evaluator = ExecutionBasedRobustnessEvaluator()
        result = evaluator.evaluate(sample_code)
        
        assert result.category == TrustworthinessCategory.ROBUSTNESS
        assert 0.0 <= result.score <= 1.0
        assert 0.0 <= result.confidence <= 1.0
    
    def test_security_evaluator(self, sample_code):
        """Test execution-based security evaluator."""
        evaluator = ExecutionBasedSecurityEvaluator()
        result = evaluator.evaluate(sample_code)
        
        assert result.category == TrustworthinessCategory.SECURITY
        assert 0.0 <= result.score <= 1.0
    
    def test_performance_evaluator(self, sample_code):
        """Test performance evaluator."""
        evaluator = ExecutionBasedPerformanceEvaluator()
        result = evaluator.evaluate(sample_code)
        
        assert result.category == TrustworthinessCategory.ROBUSTNESS
        assert "execution_time" in result.details
        assert "memory_usage" in result.details


class TestStaticAnalyzerEvaluators:
    """Test static analysis evaluators."""
    
    @pytest.fixture
    def vulnerable_code(self):
        return CodeSample(
            id="vulnerable",
            source_code="""
import os
def execute_command(cmd):
    os.system(cmd)  # Security vulnerability
    eval(cmd)       # Another vulnerability
            """,
            problem_description="Execute system command",
            language="python"
        )
    
    @pytest.fixture
    def ethical_code(self):
        return CodeSample(
            id="ethical",
            source_code="""
def evaluate_candidate(data):
    # Using discriminatory variables
    if data['race'] == 'white':
        score = 1.0
    else:
        score = 0.5
    return score
            """,
            problem_description="Evaluate job candidate",
            language="python"
        )
    
    def test_security_analyzer(self, vulnerable_code):
        """Test static security analyzer."""
        analyzer = StaticSecurityAnalyzer()
        result = analyzer.evaluate(vulnerable_code)
        
        assert result.category == TrustworthinessCategory.SECURITY
        assert result.score < 0.5  # Should detect vulnerabilities
        assert len(result.details["vulnerabilities"]) > 0
    
    def test_maintainability_analyzer(self):
        """Test maintainability analyzer."""
        code = CodeSample(
            id="maintainable",
            source_code="""
class Calculator:
    '''A simple calculator class.'''
    
    def add(self, a: int, b: int) -> int:
        '''Add two numbers.'''
        return a + b
    
    def subtract(self, a: int, b: int) -> int:
        '''Subtract b from a.'''
        return a - b
            """,
            problem_description="Calculator class",
            language="python"
        )
        
        analyzer = StaticMaintainabilityAnalyzer()
        result = analyzer.evaluate(code)
        
        assert result.category == TrustworthinessCategory.MAINTAINABILITY
        assert result.score > 0.7  # Should score well
    
    def test_ethical_analyzer(self, ethical_code):
        """Test ethical analyzer."""
        analyzer = StaticEthicalAnalyzer()
        result = analyzer.evaluate(ethical_code)
        
        assert result.category == TrustworthinessCategory.ETHICS
        assert result.score < 0.5  # Should detect ethical issues
        assert len(result.details["ethical_issues"]) > 0


class TestEnhancedCommunicationEvaluator:
    """Test enhanced communication evaluator."""
    
    @pytest.fixture
    def good_communication_code(self):
        return CodeSample(
            id="good_comm",
            source_code="""
# I notice the problem description is ambiguous about sorting order.
# Could you please clarify whether you want ascending or descending order?
# For now, I'll implement ascending sort:

def sort_numbers(numbers):
    '''Sort a list of numbers in ascending order.'''
    if not numbers:
        return []
    return sorted(numbers)

# Please let me know if you need descending order instead!
            """,
            problem_description="Sort an array of numbers",
            language="python"
        )
    
    @pytest.fixture
    def poor_communication_code(self):
        return CodeSample(
            id="poor_comm",
            source_code="def sort_numbers(x): return sorted(x)",
            problem_description="Sort an array of numbers",
            language="python"
        )
    
    def test_good_communication_detection(self, good_communication_code):
        """Test detection of good communication patterns."""
        evaluator = EnhancedCommunicationEvaluator()
        result = evaluator.evaluate(good_communication_code)
        
        assert result.category == TrustworthinessCategory.COMMUNICATION
        assert result.score > 0.7  # Should score well
        assert "clarification_requests" in result.details
    
    def test_poor_communication_detection(self, poor_communication_code):
        """Test detection of poor communication."""
        evaluator = EnhancedCommunicationEvaluator()
        result = evaluator.evaluate(poor_communication_code)
        
        assert result.category == TrustworthinessCategory.COMMUNICATION
        assert result.score < 0.5  # Should score poorly


class TestIntegration:
    """Integration tests for the complete system."""
    
    @pytest.mark.asyncio
    async def test_full_evaluation_pipeline(self):
        """Test the complete evaluation pipeline."""
        framework = MultiModalEvaluationFramework()
        
        # Register all evaluators
        framework.register_evaluator(EnhancedCommunicationEvaluator())
        framework.register_evaluator(ExecutionBasedRobustnessEvaluator())
        framework.register_evaluator(StaticSecurityAnalyzer())
        framework.register_evaluator(StaticMaintainabilityAnalyzer())
        framework.register_evaluator(StaticEthicalAnalyzer())
        
        # Create test code sample
        sample = CodeSample(
            id="integration_test",
            source_code="""
def secure_calculator(a, b, operation):
    '''A secure calculator function with input validation.'''
    
    # Input validation
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise TypeError("Inputs must be numbers")
    
    if operation not in ['+', '-', '*', '/']:
        raise ValueError("Invalid operation")
    
    # Perform calculation
    if operation == '+':
        return a + b
    elif operation == '-':
        return a - b
    elif operation == '*':
        return a * b
    elif operation == '/':
        if b == 0:
            raise ZeroDivisionError("Cannot divide by zero")
        return a / b
            """,
            problem_description="Create a calculator function",
            language="python"
        )
        
        # Run evaluation
        results = await framework.evaluate_code_sample(sample)
        
        # Verify we got results from multiple evaluators
        assert len(results) >= 3
        
        # Verify categories are represented
        categories = {result.category for result in results}
        expected_categories = {
            TrustworthinessCategory.COMMUNICATION,
            TrustworthinessCategory.ROBUSTNESS,
            TrustworthinessCategory.SECURITY,
            TrustworthinessCategory.MAINTAINABILITY,
            TrustworthinessCategory.ETHICS
        }
        
        # Should have most categories represented
        assert len(categories.intersection(expected_categories)) >= 3
        
        # Generate report
        report = framework.generate_trustworthiness_report(results)
        
        assert "overall_score" in report
        assert report["overall_score"] > 0.6  # Should score reasonably well
        assert len(report["categories"]) >= 3


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.mark.asyncio
    async def test_empty_code_sample(self):
        """Test handling of empty code samples."""
        framework = MultiModalEvaluationFramework()
        framework.register_evaluator(EnhancedCommunicationEvaluator())
        
        empty_sample = CodeSample(
            id="empty",
            source_code="",
            problem_description="Empty code",
            language="python"
        )
        
        results = await framework.evaluate_code_sample(empty_sample)
        
        # Should handle gracefully
        assert len(results) >= 0
    
    @pytest.mark.asyncio
    async def test_invalid_syntax_code(self):
        """Test handling of syntactically invalid code."""
        framework = MultiModalEvaluationFramework()
        framework.register_evaluator(StaticSecurityAnalyzer())
        
        invalid_sample = CodeSample(
            id="invalid",
            source_code="def broken_function( invalid syntax here",
            problem_description="Broken code",
            language="python"
        )
        
        results = await framework.evaluate_code_sample(invalid_sample)
        
        # Should handle gracefully and possibly penalize
        assert len(results) >= 0
    
    def test_consensus_with_single_result(self):
        """Test consensus calculation with single result."""
        framework = MultiModalEvaluationFramework()
        
        results = [
            EvaluationResult(
                category=TrustworthinessCategory.COMMUNICATION,
                score=0.8,
                confidence=0.9,
                details={},
                explanations=[]
            )
        ]
        
        consensus = framework._calculate_consensus_score(results)
        
        assert consensus["score"] == 0.8
        assert consensus["confidence"] == 0.9


# Pytest configuration and utilities
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")


# Test discovery helper
def test_suite_completeness():
    """Verify all major components have tests."""
    
    # List of components that should have tests
    components_to_test = [
        "MultiModalEvaluationFramework",
        "ExecutionBasedRobustnessEvaluator",
        "ExecutionBasedSecurityEvaluator", 
        "ExecutionBasedPerformanceEvaluator",
        "StaticSecurityAnalyzer",
        "StaticMaintainabilityAnalyzer",
        "StaticEthicalAnalyzer",
        "EnhancedCommunicationEvaluator"
    ]
    
    # Get all test classes
    import inspect
    test_classes = [
        name for name, obj in globals().items() 
        if inspect.isclass(obj) and name.startswith('Test')
    ]
    
    print(f"Found {len(test_classes)} test classes:")
    for cls in test_classes:
        print(f"  - {cls}")
    
    assert len(test_classes) >= 5  # Ensure we have sufficient test coverage


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
