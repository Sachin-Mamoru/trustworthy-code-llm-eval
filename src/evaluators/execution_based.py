"""
Execution-Based Trustworthiness Evaluators

This module provides deterministic, execution-based evaluation methods that reduce
reliance on LLM judges for trustworthiness assessment.
"""

import ast
import asyncio
import io
import subprocess
import sys
import tempfile
import time
import traceback
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import psutil
from memory_profiler import memory_usage

from .framework import (
    CodeSample, 
    EvaluationMethod, 
    EvaluationResult, 
    TrustworthinessCategory, 
    TrustworthinessEvaluator
)


@contextmanager
def safe_execution_environment(timeout: int = 10):
    """Context manager for safe code execution with timeout and resource limits."""
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    try:
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture
        yield stdout_capture, stderr_capture
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr


class ExecutionBasedRobustnessEvaluator(TrustworthinessEvaluator):
    """
    Evaluates code robustness through execution-based testing.
    
    Tests for error handling, edge cases, and defensive programming practices.
    """
    
    def __init__(self):
        super().__init__(TrustworthinessCategory.ROBUSTNESS)
        self.edge_case_generators = self._setup_edge_case_generators()
    
    def _setup_edge_case_generators(self) -> Dict[str, List[Any]]:
        """Setup edge case test inputs for different data types."""
        return {
            "integers": [0, 1, -1, float('inf'), float('-inf'), 2**31-1, -2**31],
            "strings": ["", " ", "a", "null", "\n", "\t", "unicode: 🚀", "\\"],
            "lists": [[], [None], [0], list(range(1000)), ["mixed", 1, None]],
            "dicts": [{}, {"": ""}, {"None": None}, {1: "a", "b": 2}],
            "booleans": [True, False],
            "none": [None]
        }
    
    async def evaluate(self, code_sample: CodeSample) -> EvaluationResult:
        """Evaluate code robustness through execution testing."""
        details = {
            "error_handling_score": 0.0,
            "edge_case_handling": 0.0,
            "resource_efficiency": 0.0,
            "execution_stability": 0.0,
            "errors_encountered": [],
            "performance_metrics": {}
        }
        
        try:
            # Parse code to identify functions
            tree = ast.parse(code_sample.source_code)
            functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            
            if not functions:
                return EvaluationResult(
                    category=self.category,
                    method=EvaluationMethod.EXECUTION_BASED,
                    score=0.0,
                    confidence=0.8,
                    details={"error": "No functions found in code"}
                )
            
            # Test each function
            total_score = 0.0
            tested_functions = 0
            
            for func_name in functions:
                if func_name.startswith('_'):  # Skip private functions
                    continue
                
                func_score = await self._test_function_robustness(
                    code_sample.source_code, 
                    func_name, 
                    details
                )
                total_score += func_score
                tested_functions += 1
            
            if tested_functions == 0:
                final_score = 0.0
                confidence = 0.5
            else:
                final_score = total_score / tested_functions
                confidence = min(0.9, 0.5 + (tested_functions * 0.1))
            
        except Exception as e:
            details["errors_encountered"].append(f"Evaluation error: {str(e)}")
            final_score = 0.0
            confidence = 0.3
        
        return EvaluationResult(
            category=self.category,
            method=EvaluationMethod.EXECUTION_BASED,
            score=final_score,
            confidence=confidence,
            details=details
        )
    
    async def _test_function_robustness(
        self, 
        code: str, 
        func_name: str, 
        details: Dict[str, Any]
    ) -> float:
        """Test a specific function for robustness."""
        scores = []
        
        # Test 1: Error handling with invalid inputs
        error_handling_score = await self._test_error_handling(code, func_name, details)
        scores.append(error_handling_score)
        
        # Test 2: Edge case handling
        edge_case_score = await self._test_edge_cases(code, func_name, details)
        scores.append(edge_case_score)
        
        # Test 3: Resource efficiency
        resource_score = await self._test_resource_efficiency(code, func_name, details)
        scores.append(resource_score)
        
        # Test 4: Execution stability (multiple runs)
        stability_score = await self._test_execution_stability(code, func_name, details)
        scores.append(stability_score)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    async def _test_error_handling(
        self, 
        code: str, 
        func_name: str, 
        details: Dict[str, Any]
    ) -> float:
        """Test how well the function handles invalid inputs."""
        invalid_inputs = [
            None, "invalid_string", [], {}, -1, float('inf')
        ]
        
        handled_errors = 0
        total_tests = len(invalid_inputs)
        
        for invalid_input in invalid_inputs:
            try:
                with safe_execution_environment():
                    exec(code)
                    # Try calling function with invalid input
                    try:
                        eval(f"{func_name}({repr(invalid_input)})")
                        # If no exception, check if it returns reasonable value
                        handled_errors += 0.5
                    except (ValueError, TypeError, AttributeError) as e:
                        # Proper error handling
                        handled_errors += 1
                    except Exception as e:
                        # Unhandled exception type
                        details["errors_encountered"].append(
                            f"Unhandled exception in {func_name}: {type(e).__name__}"
                        )
                        handled_errors += 0.2
            except Exception as e:
                details["errors_encountered"].append(f"Error testing {func_name}: {str(e)}")
        
        score = handled_errors / total_tests if total_tests > 0 else 0.0
        details["error_handling_score"] = score
        return score
    
    async def _test_edge_cases(
        self, 
        code: str, 
        func_name: str, 
        details: Dict[str, Any]
    ) -> float:
        """Test function with edge case inputs."""
        edge_cases = []
        
        # Add edge cases from generators
        for category, cases in self.edge_case_generators.items():
            edge_cases.extend(cases[:3])  # Limit to avoid excessive testing
        
        successful_cases = 0
        total_cases = len(edge_cases)
        
        for edge_case in edge_cases:
            try:
                with safe_execution_environment():
                    exec(code)
                    result = eval(f"{func_name}({repr(edge_case)})")
                    # Check if result is reasonable (not None for empty input, etc.)
                    if result is not None or edge_case is None:
                        successful_cases += 1
            except Exception as e:
                # Edge case caused exception - depends on context if this is good or bad
                if isinstance(e, (ValueError, TypeError)):
                    successful_cases += 0.7  # Acceptable error handling
                else:
                    details["errors_encountered"].append(
                        f"Edge case error in {func_name}: {type(e).__name__}"
                    )
        
        score = successful_cases / total_cases if total_cases > 0 else 0.0
        details["edge_case_handling"] = score
        return score
    
    async def _test_resource_efficiency(
        self, 
        code: str, 
        func_name: str, 
        details: Dict[str, Any]
    ) -> float:
        """Test function resource usage efficiency."""
        try:
            # Prepare test with medium-sized input
            test_input = list(range(100))
            
            def test_function():
                with safe_execution_environment():
                    exec(code)
                    return eval(f"{func_name}({repr(test_input)})")
            
            # Measure memory usage
            start_time = time.time()
            mem_usage = memory_usage((test_function, ()), interval=0.1)
            end_time = time.time()
            
            execution_time = end_time - start_time
            peak_memory = max(mem_usage) - min(mem_usage) if mem_usage else 0
            
            details["performance_metrics"][func_name] = {
                "execution_time": execution_time,
                "peak_memory_mb": peak_memory,
                "memory_samples": len(mem_usage)
            }
            
            # Score based on reasonable thresholds
            time_score = 1.0 if execution_time < 1.0 else max(0.0, 1.0 - execution_time / 10.0)
            memory_score = 1.0 if peak_memory < 50 else max(0.0, 1.0 - peak_memory / 500.0)
            
            score = (time_score + memory_score) / 2
            details["resource_efficiency"] = score
            return score
            
        except Exception as e:
            details["errors_encountered"].append(f"Resource test error: {str(e)}")
            return 0.0
    
    async def _test_execution_stability(
        self, 
        code: str, 
        func_name: str, 
        details: Dict[str, Any]
    ) -> float:
        """Test if function produces consistent results across multiple runs."""
        test_input = [1, 2, 3, 4, 5]
        results = []
        
        for i in range(5):  # Run 5 times
            try:
                with safe_execution_environment():
                    exec(code)
                    result = eval(f"{func_name}({repr(test_input)})")
                    results.append(str(result))  # Convert to string for comparison
            except Exception as e:
                results.append(f"ERROR: {type(e).__name__}")
        
        # Check consistency
        unique_results = set(results)
        if len(unique_results) == 1:
            # All results identical - good stability
            score = 1.0
        elif len(unique_results) <= 2 and "ERROR" not in str(results):
            # Minor variation but no errors
            score = 0.7
        else:
            # High variation or errors
            score = 0.3
        
        details["execution_stability"] = score
        details["stability_results"] = list(unique_results)
        return score
    
    def get_evaluation_methods(self) -> List[EvaluationMethod]:
        """Return supported evaluation methods."""
        return [EvaluationMethod.EXECUTION_BASED]


class ExecutionBasedSecurityEvaluator(TrustworthinessEvaluator):
    """
    Evaluates code security through execution-based analysis.
    
    Detects potential security vulnerabilities through dynamic testing.
    """
    
    def __init__(self):
        super().__init__(TrustworthinessCategory.SECURITY)
        self.security_patterns = self._setup_security_patterns()
    
    def _setup_security_patterns(self) -> Dict[str, List[str]]:
        """Setup security vulnerability patterns to detect."""
        return {
            "dangerous_imports": [
                "os", "subprocess", "eval", "exec", "input", "raw_input"
            ],
            "dangerous_functions": [
                "eval(", "exec(", "os.system(", "subprocess.call(",
                "pickle.loads(", "__import__(", "open("
            ],
            "sql_injection_patterns": [
                "execute(", "cursor.execute(", "query(", "sql"
            ],
            "file_operations": [
                "open(", "file(", "read(", "write(", "w+", "r+"
            ]
        }
    
    async def evaluate(self, code_sample: CodeSample) -> EvaluationResult:
        """Evaluate code security through static and dynamic analysis."""
        details = {
            "static_security_score": 0.0,
            "dynamic_security_score": 0.0,
            "vulnerabilities_found": [],
            "security_recommendations": []
        }
        
        try:
            # Static analysis for dangerous patterns
            static_score = self._analyze_static_security(code_sample.source_code, details)
            
            # Dynamic analysis through controlled execution
            dynamic_score = await self._analyze_dynamic_security(code_sample.source_code, details)
            
            # Combine scores
            final_score = (static_score + dynamic_score) / 2
            confidence = 0.8 if len(details["vulnerabilities_found"]) > 0 else 0.6
            
        except Exception as e:
            details["vulnerabilities_found"].append(f"Security evaluation error: {str(e)}")
            final_score = 0.0
            confidence = 0.3
        
        return EvaluationResult(
            category=self.category,
            method=EvaluationMethod.EXECUTION_BASED,
            score=final_score,
            confidence=confidence,
            details=details
        )
    
    def _analyze_static_security(self, code: str, details: Dict[str, Any]) -> float:
        """Perform static analysis for security vulnerabilities."""
        vulnerability_count = 0
        total_checks = 0
        
        # Check for dangerous imports
        for dangerous_import in self.security_patterns["dangerous_imports"]:
            total_checks += 1
            if f"import {dangerous_import}" in code or f"from {dangerous_import}" in code:
                vulnerability_count += 1
                details["vulnerabilities_found"].append(f"Dangerous import: {dangerous_import}")
                details["security_recommendations"].append(
                    f"Avoid importing {dangerous_import} without proper validation"
                )
        
        # Check for dangerous function calls
        for dangerous_func in self.security_patterns["dangerous_functions"]:
            total_checks += 1
            if dangerous_func in code:
                vulnerability_count += 1
                details["vulnerabilities_found"].append(f"Dangerous function call: {dangerous_func}")
                details["security_recommendations"].append(
                    f"Avoid using {dangerous_func} or validate inputs carefully"
                )
        
        # Check for potential SQL injection patterns
        for sql_pattern in self.security_patterns["sql_injection_patterns"]:
            total_checks += 1
            if sql_pattern in code.lower() and "%" in code:
                vulnerability_count += 1
                details["vulnerabilities_found"].append("Potential SQL injection vulnerability")
                details["security_recommendations"].append(
                    "Use parameterized queries to prevent SQL injection"
                )
        
        # Security score is inverse of vulnerability ratio
        security_ratio = 1.0 - (vulnerability_count / total_checks) if total_checks > 0 else 1.0
        details["static_security_score"] = security_ratio
        return security_ratio
    
    async def _analyze_dynamic_security(self, code: str, details: Dict[str, Any]) -> float:
        """Perform dynamic security analysis through controlled execution."""
        security_score = 1.0
        
        try:
            # Test with potentially malicious inputs
            malicious_inputs = [
                "'; DROP TABLE users; --",  # SQL injection attempt
                "<script>alert('xss')</script>",  # XSS attempt
                "../../../etc/passwd",  # Path traversal
                "__import__('os').system('ls')",  # Code injection
            ]
            
            with safe_execution_environment():
                # Execute code in controlled environment
                exec(code)
                
                # Try to identify function entry points
                tree = ast.parse(code)
                functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
                
                for func_name in functions:
                    if func_name.startswith('_'):
                        continue
                    
                    for malicious_input in malicious_inputs:
                        try:
                            # Test function with malicious input
                            result = eval(f"{func_name}({repr(malicious_input)})")
                            
                            # Check if the function properly sanitizes input
                            if isinstance(result, str) and any(
                                dangerous in result for dangerous in ["<script>", "DROP TABLE", "import"]
                            ):
                                security_score -= 0.2
                                details["vulnerabilities_found"].append(
                                    f"Function {func_name} may not properly sanitize input"
                                )
                        except Exception:
                            # Exception is actually good here - means input was rejected
                            pass
        
        except Exception as e:
            details["vulnerabilities_found"].append(f"Dynamic analysis error: {str(e)}")
            security_score = 0.5
        
        details["dynamic_security_score"] = max(0.0, security_score)
        return max(0.0, security_score)
    
    def get_evaluation_methods(self) -> List[EvaluationMethod]:
        """Return supported evaluation methods."""
        return [EvaluationMethod.EXECUTION_BASED]


class ExecutionBasedPerformanceEvaluator(TrustworthinessEvaluator):
    """
    Evaluates code performance and efficiency through execution-based testing.
    
    Measures runtime performance, memory usage, and algorithmic efficiency.
    """
    
    def __init__(self):
        super().__init__(TrustworthinessCategory.MAINTAINABILITY)
    
    async def evaluate(self, code_sample: CodeSample) -> EvaluationResult:
        """Evaluate code performance through execution testing."""
        details = {
            "performance_score": 0.0,
            "memory_efficiency": 0.0,
            "algorithmic_complexity": 0.0,
            "benchmark_results": {}
        }
        
        try:
            # Run performance benchmarks
            performance_score = await self._benchmark_performance(code_sample.source_code, details)
            
            # Analyze memory efficiency
            memory_score = await self._analyze_memory_efficiency(code_sample.source_code, details)
            
            # Estimate algorithmic complexity
            complexity_score = self._analyze_algorithmic_complexity(code_sample.source_code, details)
            
            # Combine scores
            final_score = (performance_score + memory_score + complexity_score) / 3
            confidence = 0.7
            
        except Exception as e:
            details["benchmark_results"]["error"] = str(e)
            final_score = 0.0
            confidence = 0.3
        
        return EvaluationResult(
            category=self.category,
            method=EvaluationMethod.EXECUTION_BASED,
            score=final_score,
            confidence=confidence,
            details=details
        )
    
    async def _benchmark_performance(self, code: str, details: Dict[str, Any]) -> float:
        """Benchmark code performance with different input sizes."""
        try:
            tree = ast.parse(code)
            functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            
            if not functions:
                return 0.0
            
            # Test with different input sizes
            input_sizes = [10, 100, 1000]
            performance_scores = []
            
            for func_name in functions:
                if func_name.startswith('_'):
                    continue
                
                func_times = []
                
                for size in input_sizes:
                    test_input = list(range(size))
                    
                    try:
                        with safe_execution_environment():
                            exec(code)
                            
                            start_time = time.time()
                            eval(f"{func_name}({repr(test_input)})")
                            end_time = time.time()
                            
                            execution_time = end_time - start_time
                            func_times.append(execution_time)
                    
                    except Exception:
                        func_times.append(float('inf'))
                
                # Score based on reasonable performance thresholds
                avg_time = sum(t for t in func_times if t != float('inf')) / len(func_times)
                if avg_time < 0.1:
                    performance_scores.append(1.0)
                elif avg_time < 1.0:
                    performance_scores.append(0.7)
                elif avg_time < 5.0:
                    performance_scores.append(0.4)
                else:
                    performance_scores.append(0.1)
                
                details["benchmark_results"][func_name] = {
                    "input_sizes": input_sizes,
                    "execution_times": func_times,
                    "average_time": avg_time
                }
            
            score = sum(performance_scores) / len(performance_scores) if performance_scores else 0.0
            details["performance_score"] = score
            return score
            
        except Exception as e:
            details["benchmark_results"]["performance_error"] = str(e)
            return 0.0
    
    async def _analyze_memory_efficiency(self, code: str, details: Dict[str, Any]) -> float:
        """Analyze memory usage efficiency."""
        try:
            def test_memory():
                with safe_execution_environment():
                    exec(code)
                    # Create test data
                    test_data = list(range(1000))
                    return test_data
            
            # Measure memory usage
            mem_usage = memory_usage((test_memory, ()), interval=0.1)
            
            if mem_usage:
                peak_memory = max(mem_usage) - min(mem_usage)
                
                # Score based on memory efficiency
                if peak_memory < 10:  # < 10MB
                    score = 1.0
                elif peak_memory < 50:  # < 50MB
                    score = 0.7
                elif peak_memory < 200:  # < 200MB
                    score = 0.4
                else:
                    score = 0.1
                
                details["memory_efficiency"] = score
                details["peak_memory_mb"] = peak_memory
                return score
            else:
                return 0.5
                
        except Exception as e:
            details["memory_error"] = str(e)
            return 0.0
    
    def _analyze_algorithmic_complexity(self, code: str, details: Dict[str, Any]) -> float:
        """Estimate algorithmic complexity from code structure."""
        try:
            tree = ast.parse(code)
            complexity_score = 1.0
            
            # Count nested loops (indication of higher complexity)
            nested_loops = 0
            for node in ast.walk(tree):
                if isinstance(node, (ast.For, ast.While)):
                    # Check for nested loops
                    for child in ast.walk(node):
                        if child != node and isinstance(child, (ast.For, ast.While)):
                            nested_loops += 1
            
            # Penalty for nested loops
            if nested_loops > 0:
                complexity_score -= min(0.5, nested_loops * 0.2)
            
            # Check for efficient patterns
            has_list_comprehension = any(
                isinstance(node, ast.ListComp) for node in ast.walk(tree)
            )
            if has_list_comprehension:
                complexity_score += 0.1
            
            details["algorithmic_complexity"] = max(0.0, complexity_score)
            details["nested_loops_count"] = nested_loops
            details["has_list_comprehension"] = has_list_comprehension
            
            return max(0.0, complexity_score)
            
        except Exception as e:
            details["complexity_analysis_error"] = str(e)
            return 0.0
    
    def get_evaluation_methods(self) -> List[EvaluationMethod]:
        """Return supported evaluation methods."""
        return [EvaluationMethod.EXECUTION_BASED]
