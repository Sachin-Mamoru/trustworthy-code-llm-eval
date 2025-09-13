"""
Comprehensive test suite for the TrustworthyCodeLLM evaluation framework.

This module provides extensive testing coverage for all framework components,
including unit tests, integration tests, and performance benchmarks.
"""

import asyncio
import pytest
import tempfile
from pathlib import Path
import json
import time
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any

# Test framework components
from src.framework import MultiModalEvaluationFramework, CodeSample, EvaluationResult, TrustworthinessCategory, EvaluationMethod


class TestCodeSample:
    """Test the CodeSample data structure."""
    
    def test_code_sample_creation(self):
        """Test creating a basic code sample."""
        sample = CodeSample(
            id="test_1",
            source_code="def hello(): return 'world'",
            problem_description="Simple hello function",
            language="python"
        )
        
        assert sample.id == "test_1"
        assert "hello" in sample.source_code
        assert sample.language == "python"
        assert sample.metadata == {}
    
    def test_code_sample_with_metadata(self):
        """Test code sample with additional metadata."""
        metadata = {"author": "test", "difficulty": "easy"}
        sample = CodeSample(
            id="test_2",
            source_code="x = 42",
            problem_description="Variable assignment",
            language="python",
            metadata=metadata
        )
        
        assert sample.metadata["author"] == "test"
        assert sample.metadata["difficulty"] == "easy"
    
    def test_code_sample_validation(self):
        """Test code sample validation."""
        # Valid sample
        sample = CodeSample(
            id="valid",
            source_code="print('hello')",
            problem_description="Print statement",
            language="python"
        )
        
        assert sample.id is not None
        assert len(sample.source_code) > 0


class TestEvaluationResult:
    """Test the EvaluationResult data structure."""
    
    def test_evaluation_result_creation(self):
        """Test creating an evaluation result."""
        result = EvaluationResult(
            category=TrustworthinessCategory.SECURITY,
            score=0.85,
            confidence=0.9,
            method=EvaluationMethod.STATIC_ANALYSIS,
            details={"issues": 2, "severity": "low"}
        )
        
        assert result.category == TrustworthinessCategory.SECURITY
        assert result.score == 0.85
        assert result.confidence == 0.9
        assert result.method == EvaluationMethod.STATIC_ANALYSIS
        assert result.details["issues"] == 2
    
    def test_result_validation(self):
        """Test evaluation result validation."""
        # Valid result
        result = EvaluationResult(
            category=TrustworthinessCategory.ROBUSTNESS,
            score=0.5,
            confidence=0.7,
            method=EvaluationMethod.EXECUTION_BASED
        )
        
        assert 0 <= result.score <= 1
        assert 0 <= result.confidence <= 1


class TestMultiModalFramework:
    """Test the main evaluation framework."""
    
    @pytest.fixture
    def framework(self):
        """Create a framework instance for testing."""
        return MultiModalEvaluationFramework()
    
    @pytest.fixture
    def mock_evaluator(self):
        """Create a mock evaluator for testing."""
        evaluator = Mock()
        evaluator.get_supported_categories.return_value = [TrustworthinessCategory.SECURITY]
        evaluator.evaluate.return_value = EvaluationResult(
            category=TrustworthinessCategory.SECURITY,
            score=0.8,
            confidence=0.9,
            method=EvaluationMethod.STATIC_ANALYSIS
        )
        return evaluator
    
    def test_framework_initialization(self, framework):
        """Test framework initialization."""
        assert framework is not None
        assert len(framework.evaluators) == 0
    
    def test_register_evaluator(self, framework, mock_evaluator):
        """Test registering an evaluator."""
        framework.register_evaluator(mock_evaluator)
        assert len(framework.evaluators) == 1
        assert mock_evaluator in framework.evaluators
    
    @pytest.mark.asyncio
    async def test_evaluate_code_sample(self, framework, mock_evaluator):
        """Test evaluating a code sample."""
        framework.register_evaluator(mock_evaluator)
        
        sample = CodeSample(
            id="test",
            source_code="def test(): pass",
            problem_description="Test function",
            language="python"
        )
        
        results = await framework.evaluate_code_sample(sample)
        
        assert len(results) == 1
        assert results[0].category == TrustworthinessCategory.SECURITY
        assert results[0].score == 0.8
    
    @pytest.mark.asyncio
    async def test_evaluate_multiple_samples(self, framework, mock_evaluator):
        """Test batch evaluation of multiple samples."""
        framework.register_evaluator(mock_evaluator)
        
        samples = [
            CodeSample(id=f"test_{i}", source_code=f"def func_{i}(): pass", 
                      problem_description=f"Function {i}", language="python")
            for i in range(3)
        ]
        
        results = await framework.evaluate_samples(samples)
        
        assert len(results) == 3
        for sample_results in results:
            assert len(sample_results) == 1
            assert sample_results[0].category == TrustworthinessCategory.SECURITY


class TestEvaluatorIntegration:
    """Integration tests for evaluators."""
    
    @pytest.fixture
    def framework_with_evaluators(self):
        """Create framework with actual evaluators."""
        from src.evaluators.enhanced_communication import EnhancedCommunicationEvaluator
        from src.evaluators.static_analysis import StaticSecurityAnalyzer
        
        framework = MultiModalEvaluationFramework()
        framework.register_evaluator(EnhancedCommunicationEvaluator())
        framework.register_evaluator(StaticSecurityAnalyzer())
        return framework
    
    @pytest.mark.asyncio
    async def test_real_evaluators(self, framework_with_evaluators):
        """Test with real evaluators."""
        sample = CodeSample(
            id="integration_test",
            source_code="""
def process_user_input(user_input):
    # This function processes user input safely
    if not user_input:
        return None
    
    # Validate input length
    if len(user_input) > 1000:
        raise ValueError("Input too long")
    
    # Basic sanitization
    cleaned = user_input.strip()
    return cleaned.lower()
""",
            problem_description="Process and sanitize user input safely",
            language="python"
        )
        
        results = await framework_with_evaluators.evaluate_code_sample(sample)
        
        # Should have results from multiple evaluators
        assert len(results) >= 1
        
        # Check that we have different categories
        categories = {result.category for result in results}
        assert len(categories) >= 1


class TestPerformance:
    """Performance and scalability tests."""
    
    @pytest.mark.asyncio
    async def test_evaluation_performance(self):
        """Test evaluation performance with timing."""
        from src.evaluators.enhanced_communication import EnhancedCommunicationEvaluator
        
        evaluator = EnhancedCommunicationEvaluator()
        sample = CodeSample(
            id="perf_test",
            source_code="def simple_function(): return 42",
            problem_description="Simple function",
            language="python"
        )
        
        start_time = time.time()
        result = await evaluator.evaluate(sample)
        end_time = time.time()
        
        evaluation_time = end_time - start_time
        
        # Evaluation should complete within reasonable time
        assert evaluation_time < 30.0  # 30 seconds max
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_batch_performance(self):
        """Test batch evaluation performance."""
        framework = MultiModalEvaluationFramework()
        
        # Create multiple samples
        samples = [
            CodeSample(
                id=f"batch_test_{i}",
                source_code=f"def func_{i}(): return {i}",
                problem_description=f"Function {i}",
                language="python"
            )
            for i in range(10)
        ]
        
        start_time = time.time()
        results = await framework.evaluate_samples(samples)
        end_time = time.time()
        
        batch_time = end_time - start_time
        
        # Batch processing should be efficient
        assert batch_time < 60.0  # 1 minute max for 10 samples
        assert len(results) == 10


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_invalid_code_handling(self):
        """Test handling of invalid code."""
        from src.evaluators.enhanced_communication import EnhancedCommunicationEvaluator
        
        evaluator = EnhancedCommunicationEvaluator()
        sample = CodeSample(
            id="invalid_test",
            source_code="def invalid_syntax( missing_paren:",
            problem_description="Syntactically invalid code",
            language="python"
        )
        
        # Should handle gracefully without crashing
        result = await evaluator.evaluate(sample)
        
        # Should still return a result, possibly with lower confidence
        assert result is not None
        assert isinstance(result.score, (int, float))
        assert 0 <= result.score <= 1
    
    @pytest.mark.asyncio
    async def test_empty_code_handling(self):
        """Test handling of empty code."""
        framework = MultiModalEvaluationFramework()
        
        sample = CodeSample(
            id="empty_test",
            source_code="",
            problem_description="Empty code",
            language="python"
        )
        
        results = await framework.evaluate_code_sample(sample)
        
        # Should handle gracefully
        assert isinstance(results, list)
    
    def test_malformed_sample(self):
        """Test handling of malformed samples."""
        # Test with None values
        sample = CodeSample(
            id="malformed",
            source_code=None,
            problem_description="",
            language="python"
        )
        
        # Should create sample but source_code handling depends on evaluator
        assert sample.id == "malformed"


class TestCLIIntegration:
    """Test CLI functionality."""
    
    def test_cli_import(self):
        """Test that CLI can be imported."""
        try:
            from cli import TrustworthyCodeCLI
            cli = TrustworthyCodeCLI()
            assert cli is not None
        except ImportError as e:
            pytest.skip(f"CLI dependencies not available: {e}")
    
    @pytest.mark.asyncio
    async def test_cli_evaluation(self):
        """Test CLI evaluation functionality."""
        try:
            from cli import TrustworthyCodeCLI
            
            cli = TrustworthyCodeCLI()
            
            # Test simple evaluation
            code = "def hello(): return 'world'"
            
            # This should not raise an exception
            await cli.evaluate_code(code, "Test function", "json")
            
        except ImportError as e:
            pytest.skip(f"CLI dependencies not available: {e}")


class TestReproducibility:
    """Test reproducibility of results."""
    
    @pytest.mark.asyncio
    async def test_reproducible_evaluation(self):
        """Test that evaluations are reproducible."""
        from src.evaluators.enhanced_communication import EnhancedCommunicationEvaluator
        
        evaluator = EnhancedCommunicationEvaluator()
        sample = CodeSample(
            id="repro_test",
            source_code="""
def calculate_average(numbers):
    '''Calculate the average of a list of numbers.'''
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)
""",
            problem_description="Calculate average of numbers",
            language="python"
        )
        
        # Run evaluation multiple times
        results = []
        for _ in range(3):
            result = await evaluator.evaluate(sample)
            results.append(result)
        
        # Results should be consistent (allowing for small variations due to LLM randomness)
        scores = [r.score for r in results]
        score_variance = max(scores) - min(scores)
        
        # Variance should be small for deterministic components
        assert score_variance < 0.3  # Allow for some LLM variability


class TestDatasetGeneration:
    """Test dataset generation functionality."""
    
    def test_dataset_generator_import(self):
        """Test that dataset generator can be imported."""
        try:
            from src.datasets.enhanced_humaneval_comm import EnhancedHumanEvalCommGenerator
            generator = EnhancedHumanEvalCommGenerator()
            assert generator is not None
        except ImportError as e:
            pytest.skip(f"Dataset dependencies not available: {e}")
    
    @pytest.mark.asyncio
    async def test_dataset_generation(self):
        """Test basic dataset generation."""
        try:
            from src.datasets.enhanced_humaneval_comm import EnhancedHumanEvalCommGenerator
            
            generator = EnhancedHumanEvalCommGenerator()
            
            # Generate a small sample
            samples = await generator.generate_samples(
                num_samples=2,
                categories=[TrustworthinessCategory.SECURITY]
            )
            
            assert len(samples) <= 2  # May generate fewer if limited by available data
            for sample in samples:
                assert isinstance(sample, CodeSample)
                assert sample.source_code is not None
                assert len(sample.source_code) > 0
                
        except ImportError as e:
            pytest.skip(f"Dataset dependencies not available: {e}")


class TestWebDashboard:
    """Test web dashboard functionality."""
    
    def test_dashboard_import(self):
        """Test that dashboard can be imported."""
        try:
            from web_dashboard.app import app
            assert app is not None
        except ImportError as e:
            pytest.skip(f"Dashboard dependencies not available: {e}")
    
    @pytest.mark.asyncio
    async def test_dashboard_health_endpoint(self):
        """Test dashboard health endpoint."""
        try:
            from fastapi.testclient import TestClient
            from web_dashboard.app import app
            
            client = TestClient(app)
            response = client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            
        except ImportError as e:
            pytest.skip(f"Dashboard dependencies not available: {e}")


# Performance benchmark utilities
def run_performance_benchmark():
    """Run a comprehensive performance benchmark."""
    
    async def benchmark_suite():
        """Async benchmark suite."""
        print("🔥 Running TrustworthyCodeLLM Performance Benchmark")
        print("=" * 60)
        
        # Test framework initialization
        start = time.time()
        framework = MultiModalEvaluationFramework()
        init_time = time.time() - start
        print(f"Framework initialization: {init_time:.3f}s")
        
        # Test single evaluation
        sample = CodeSample(
            id="benchmark",
            source_code="""
def secure_hash_password(password, salt):
    '''Securely hash a password with salt.'''
    import hashlib
    import hmac
    
    if not password or not salt:
        raise ValueError("Password and salt are required")
    
    # Use PBKDF2 for secure hashing
    hashed = hashlib.pbkdf2_hmac('sha256', 
                                password.encode('utf-8'), 
                                salt.encode('utf-8'), 
                                100000)
    return hashed.hex()
""",
            problem_description="Secure password hashing function",
            language="python"
        )
        
        start = time.time()
        results = await framework.evaluate_code_sample(sample)
        eval_time = time.time() - start
        print(f"Single evaluation: {eval_time:.3f}s")
        
        # Test batch evaluation
        samples = [sample] * 5
        start = time.time()
        batch_results = await framework.evaluate_samples(samples)
        batch_time = time.time() - start
        print(f"Batch evaluation (5 samples): {batch_time:.3f}s")
        print(f"Average per sample: {batch_time/5:.3f}s")
        
        print("=" * 60)
        print("✅ Performance benchmark completed")
    
    asyncio.run(benchmark_suite())


if __name__ == "__main__":
    # Run performance benchmark if called directly
    run_performance_benchmark()
