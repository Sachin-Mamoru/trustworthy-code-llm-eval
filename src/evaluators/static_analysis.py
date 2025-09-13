"""
Static Analysis Evaluators

This module provides static analysis-based evaluation for various
trustworthiness dimensions without relying on code execution.
"""

import ast
import re
from typing import Any, Dict, List, Optional, Set, Tuple
from pathlib import Path

from ..framework import (
    CodeSample, 
    EvaluationMethod, 
    EvaluationResult, 
    TrustworthinessCategory, 
    TrustworthinessEvaluator
)


class StaticSecurityAnalyzer(TrustworthinessEvaluator):
    """
    Static analysis for security vulnerabilities and patterns.
    """
    
    def __init__(self):
        super().__init__(TrustworthinessCategory.SECURITY)
        self.vulnerability_patterns = self._setup_vulnerability_patterns()
        self.secure_patterns = self._setup_secure_patterns()
    
    def _setup_vulnerability_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Setup patterns for detecting security vulnerabilities."""
        return {
            "code_injection": {
                "patterns": [
                    r"eval\s*\(",
                    r"exec\s*\(",
                    r"__import__\s*\(",
                    r"compile\s*\("
                ],
                "severity": "high",
                "description": "Code injection vulnerabilities"
            },
            "command_injection": {
                "patterns": [
                    r"os\.system\s*\(",
                    r"subprocess\.call\s*\(",
                    r"subprocess\.run\s*\(",
                    r"subprocess\.Popen\s*\("
                ],
                "severity": "high",
                "description": "Command injection vulnerabilities"
            },
            "file_traversal": {
                "patterns": [
                    r"open\s*\(\s*[^,)]*\.\./",
                    r"file\s*\(\s*[^,)]*\.\./",
                    r"\.\./"
                ],
                "severity": "medium",
                "description": "Potential path traversal vulnerabilities"
            },
            "hardcoded_secrets": {
                "patterns": [
                    r"password\s*=\s*['\"][^'\"]+['\"]",
                    r"secret\s*=\s*['\"][^'\"]+['\"]",
                    r"api_key\s*=\s*['\"][^'\"]+['\"]",
                    r"token\s*=\s*['\"][^'\"]+['\"]"
                ],
                "severity": "high",
                "description": "Hardcoded secrets or credentials"
            },
            "unsafe_deserialization": {
                "patterns": [
                    r"pickle\.loads\s*\(",
                    r"pickle\.load\s*\(",
                    r"cPickle\.loads\s*\(",
                    r"yaml\.load\s*\("
                ],
                "severity": "high",
                "description": "Unsafe deserialization"
            },
            "sql_injection": {
                "patterns": [
                    r"execute\s*\(\s*['\"].*%.*['\"]",
                    r"query\s*\(\s*['\"].*\+.*['\"]",
                    r"cursor\.execute\s*\([^)]*%[^)]*\)"
                ],
                "severity": "high",
                "description": "Potential SQL injection"
            }
        }
    
    def _setup_secure_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Setup patterns for detecting secure coding practices."""
        return {
            "input_validation": {
                "patterns": [
                    r"isinstance\s*\(",
                    r"type\s*\(",
                    r"assert\s+",
                    r"if.*not.*:",
                    r"raise\s+ValueError",
                    r"raise\s+TypeError"
                ],
                "bonus": 0.1,
                "description": "Input validation practices"
            },
            "error_handling": {
                "patterns": [
                    r"try\s*:",
                    r"except\s+\w+",
                    r"finally\s*:",
                    r"raise\s+\w+"
                ],
                "bonus": 0.1,
                "description": "Proper error handling"
            },
            "secure_random": {
                "patterns": [
                    r"secrets\.",
                    r"os\.urandom",
                    r"random\.SystemRandom"
                ],
                "bonus": 0.05,
                "description": "Cryptographically secure random generation"
            }
        }
    
    async def evaluate(self, code_sample: CodeSample) -> EvaluationResult:
        """Evaluate security through static analysis."""
        details = {
            "vulnerabilities": [],
            "secure_practices": [],
            "security_score_breakdown": {},
            "recommendations": []
        }
        
        try:
            # Analyze vulnerabilities
            vulnerability_score = self._analyze_vulnerabilities(code_sample.source_code, details)
            
            # Analyze secure practices
            secure_practices_score = self._analyze_secure_practices(code_sample.source_code, details)
            
            # AST-based analysis
            ast_score = self._analyze_ast_security(code_sample.source_code, details)
            
            # Combine scores
            final_score = (vulnerability_score + secure_practices_score + ast_score) / 3
            confidence = 0.8
            
        except Exception as e:
            details["analysis_error"] = str(e)
            final_score = 0.0
            confidence = 0.3
        
        return EvaluationResult(
            category=self.category,
            method=EvaluationMethod.STATIC_ANALYSIS,
            score=final_score,
            confidence=confidence,
            details=details
        )
    
    def _analyze_vulnerabilities(self, code: str, details: Dict[str, Any]) -> float:
        """Analyze code for security vulnerabilities."""
        vulnerability_count = 0
        total_severity_score = 0
        
        for vuln_type, vuln_info in self.vulnerability_patterns.items():
            for pattern in vuln_info["patterns"]:
                matches = re.findall(pattern, code, re.IGNORECASE)
                if matches:
                    severity_weight = {"high": 1.0, "medium": 0.6, "low": 0.3}[vuln_info["severity"]]
                    vulnerability_count += len(matches)
                    total_severity_score += len(matches) * severity_weight
                    
                    details["vulnerabilities"].append({
                        "type": vuln_type,
                        "description": vuln_info["description"],
                        "severity": vuln_info["severity"],
                        "matches": len(matches),
                        "pattern": pattern
                    })
                    
                    # Add recommendations
                    details["recommendations"].append(
                        f"Address {vuln_type}: {vuln_info['description']}"
                    )
        
        # Calculate vulnerability score (lower is better, so invert)
        if vulnerability_count == 0:
            vuln_score = 1.0
        else:
            # Penalize based on severity-weighted count
            penalty = min(1.0, total_severity_score * 0.2)
            vuln_score = max(0.0, 1.0 - penalty)
        
        details["security_score_breakdown"]["vulnerability_score"] = vuln_score
        details["security_score_breakdown"]["vulnerability_count"] = vulnerability_count
        
        return vuln_score
    
    def _analyze_secure_practices(self, code: str, details: Dict[str, Any]) -> float:
        """Analyze code for secure coding practices."""
        secure_practices_score = 0.5  # Base score
        
        for practice_type, practice_info in self.secure_patterns.items():
            for pattern in practice_info["patterns"]:
                matches = re.findall(pattern, code, re.IGNORECASE)
                if matches:
                    secure_practices_score += practice_info["bonus"]
                    
                    details["secure_practices"].append({
                        "type": practice_type,
                        "description": practice_info["description"],
                        "matches": len(matches),
                        "bonus": practice_info["bonus"]
                    })
        
        # Cap the score at 1.0
        secure_practices_score = min(1.0, secure_practices_score)
        
        details["security_score_breakdown"]["secure_practices_score"] = secure_practices_score
        
        return secure_practices_score
    
    def _analyze_ast_security(self, code: str, details: Dict[str, Any]) -> float:
        """Perform AST-based security analysis."""
        try:
            tree = ast.parse(code)
            ast_score = 1.0
            
            # Check for dangerous function calls
            dangerous_calls = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        func_name = node.func.id
                        if func_name in ["eval", "exec", "compile"]:
                            dangerous_calls.append(func_name)
                            ast_score -= 0.3
                    
                    elif isinstance(node.func, ast.Attribute):
                        if node.func.attr in ["system", "call", "run", "loads"]:
                            dangerous_calls.append(node.func.attr)
                            ast_score -= 0.2
            
            # Check for hardcoded strings that might be secrets
            string_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.Str)]
            suspicious_strings = []
            
            for string_node in string_nodes:
                value = string_node.s
                if len(value) > 20 and any(char.isdigit() for char in value) and any(char.isalpha() for char in value):
                    # Might be an encoded secret
                    suspicious_strings.append(value[:20] + "...")
                    ast_score -= 0.1
            
            details["security_score_breakdown"]["ast_analysis"] = {
                "dangerous_calls": dangerous_calls,
                "suspicious_strings_count": len(suspicious_strings),
                "ast_score": max(0.0, ast_score)
            }
            
            return max(0.0, ast_score)
            
        except SyntaxError:
            details["security_score_breakdown"]["ast_analysis"] = {"error": "Syntax error in code"}
            return 0.5
    
    def get_evaluation_methods(self) -> List[EvaluationMethod]:
        """Return supported evaluation methods."""
        return [EvaluationMethod.STATIC_ANALYSIS]


class StaticMaintainabilityAnalyzer(TrustworthinessEvaluator):
    """
    Static analysis for code maintainability and quality.
    """
    
    def __init__(self):
        super().__init__(TrustworthinessCategory.MAINTAINABILITY)
        self.quality_metrics = self._setup_quality_metrics()
    
    def _setup_quality_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Setup metrics for code quality assessment."""
        return {
            "documentation": {
                "patterns": [
                    r'""".*?"""',  # Docstrings
                    r"'''.*?'''",
                    r"#.*",  # Comments
                ],
                "weight": 0.2,
                "description": "Code documentation quality"
            },
            "naming_conventions": {
                "patterns": [
                    r"\bdef\s+[a-z_][a-z0-9_]*\s*\(",  # Function names
                    r"\bclass\s+[A-Z][a-zA-Z0-9]*\s*[\(:]",  # Class names
                    r"\b[a-z_][a-z0-9_]*\s*="  # Variable names
                ],
                "weight": 0.15,
                "description": "Naming convention adherence"
            },
            "modularity": {
                "patterns": [
                    r"\bdef\s+",  # Function definitions
                    r"\bclass\s+",  # Class definitions
                    r"\bimport\s+",  # Import statements
                    r"\bfrom\s+\w+\s+import"
                ],
                "weight": 0.2,
                "description": "Code modularity and structure"
            },
            "error_handling": {
                "patterns": [
                    r"\btry\s*:",
                    r"\bexcept\s+",
                    r"\bfinally\s*:",
                    r"\braise\s+"
                ],
                "weight": 0.15,
                "description": "Error handling implementation"
            }
        }
    
    async def evaluate(self, code_sample: CodeSample) -> EvaluationResult:
        """Evaluate maintainability through static analysis."""
        details = {
            "quality_metrics": {},
            "complexity_analysis": {},
            "structure_analysis": {},
            "recommendations": []
        }
        
        try:
            # Analyze code quality metrics
            quality_score = self._analyze_quality_metrics(code_sample.source_code, details)
            
            # Analyze code complexity
            complexity_score = self._analyze_complexity(code_sample.source_code, details)
            
            # Analyze code structure
            structure_score = self._analyze_structure(code_sample.source_code, details)
            
            # Combine scores
            final_score = (quality_score + complexity_score + structure_score) / 3
            confidence = 0.75
            
        except Exception as e:
            details["analysis_error"] = str(e)
            final_score = 0.0
            confidence = 0.3
        
        return EvaluationResult(
            category=self.category,
            method=EvaluationMethod.STATIC_ANALYSIS,
            score=final_score,
            confidence=confidence,
            details=details
        )
    
    def _analyze_quality_metrics(self, code: str, details: Dict[str, Any]) -> float:
        """Analyze code quality metrics."""
        total_score = 0.0
        total_weight = 0.0
        
        lines = code.split('\n')
        total_lines = len([line for line in lines if line.strip()])
        
        for metric_name, metric_info in self.quality_metrics.items():
            metric_score = 0.0
            
            if metric_name == "documentation":
                # Calculate documentation ratio
                doc_lines = 0
                for pattern in metric_info["patterns"]:
                    matches = re.findall(pattern, code, re.DOTALL)
                    for match in matches:
                        doc_lines += len(match.split('\n'))
                
                if total_lines > 0:
                    doc_ratio = doc_lines / total_lines
                    metric_score = min(1.0, doc_ratio * 3)  # Good if 33%+ documentation
                
            elif metric_name == "naming_conventions":
                # Check naming convention adherence
                good_names = 0
                total_names = 0
                
                for pattern in metric_info["patterns"]:
                    matches = re.findall(pattern, code)
                    total_names += len(matches)
                    good_names += len(matches)  # Assume pattern matches are good
                
                if total_names > 0:
                    metric_score = good_names / total_names
                else:
                    metric_score = 1.0  # No names to check
            
            elif metric_name == "modularity":
                # Check for modular structure
                functions = len(re.findall(r"\bdef\s+", code))
                classes = len(re.findall(r"\bclass\s+", code))
                imports = len(re.findall(r"\bimport\s+", code))
                
                # Score based on presence of modular elements
                if total_lines > 0:
                    modularity_ratio = (functions + classes + imports) / (total_lines / 10)
                    metric_score = min(1.0, modularity_ratio)
                else:
                    metric_score = 0.0
            
            elif metric_name == "error_handling":
                # Check for error handling
                error_handling_count = 0
                for pattern in metric_info["patterns"]:
                    error_handling_count += len(re.findall(pattern, code))
                
                # Score based on error handling density
                if total_lines > 0:
                    error_handling_ratio = error_handling_count / (total_lines / 20)
                    metric_score = min(1.0, error_handling_ratio)
                else:
                    metric_score = 0.0
            
            details["quality_metrics"][metric_name] = {
                "score": metric_score,
                "weight": metric_info["weight"],
                "description": metric_info["description"]
            }
            
            total_score += metric_score * metric_info["weight"]
            total_weight += metric_info["weight"]
        
        final_quality_score = total_score / total_weight if total_weight > 0 else 0.0
        return final_quality_score
    
    def _analyze_complexity(self, code: str, details: Dict[str, Any]) -> float:
        """Analyze code complexity."""
        try:
            tree = ast.parse(code)
            
            # Calculate cyclomatic complexity
            complexity = 1  # Base complexity
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.While, ast.For, ast.Try)):
                    complexity += 1
                elif isinstance(node, ast.BoolOp):
                    complexity += len(node.values) - 1
            
            # Calculate nesting depth
            max_depth = self._calculate_nesting_depth(tree)
            
            # Calculate function complexity
            function_complexities = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_complexity = self._calculate_function_complexity(node)
                    function_complexities.append(func_complexity)
            
            # Score based on complexity thresholds
            complexity_score = 1.0
            
            if complexity > 20:
                complexity_score -= 0.4
            elif complexity > 10:
                complexity_score -= 0.2
            
            if max_depth > 5:
                complexity_score -= 0.3
            elif max_depth > 3:
                complexity_score -= 0.1
            
            if function_complexities:
                avg_func_complexity = sum(function_complexities) / len(function_complexities)
                if avg_func_complexity > 10:
                    complexity_score -= 0.3
                elif avg_func_complexity > 5:
                    complexity_score -= 0.1
            
            details["complexity_analysis"] = {
                "cyclomatic_complexity": complexity,
                "max_nesting_depth": max_depth,
                "average_function_complexity": sum(function_complexities) / len(function_complexities) if function_complexities else 0,
                "complexity_score": max(0.0, complexity_score)
            }
            
            return max(0.0, complexity_score)
            
        except SyntaxError:
            details["complexity_analysis"] = {"error": "Syntax error in code"}
            return 0.5
    
    def _calculate_nesting_depth(self, node: ast.AST, current_depth: int = 0) -> int:
        """Calculate maximum nesting depth in AST."""
        max_depth = current_depth
        
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
                child_depth = self._calculate_nesting_depth(child, current_depth + 1)
                max_depth = max(max_depth, child_depth)
            else:
                child_depth = self._calculate_nesting_depth(child, current_depth)
                max_depth = max(max_depth, child_depth)
        
        return max_depth
    
    def _calculate_function_complexity(self, func_node: ast.FunctionDef) -> int:
        """Calculate complexity of a single function."""
        complexity = 1
        
        for node in ast.walk(func_node):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.Try)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        
        return complexity
    
    def _analyze_structure(self, code: str, details: Dict[str, Any]) -> float:
        """Analyze code structure and organization."""
        lines = code.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        structure_score = 1.0
        
        # Check for imports at the top
        import_lines = []
        for i, line in enumerate(lines):
            if re.match(r'^\s*(import|from)\s+', line):
                import_lines.append(i)
        
        if import_lines:
            # Check if imports are at the top (first 20% of file)
            first_import = min(import_lines)
            if first_import > len(lines) * 0.2:
                structure_score -= 0.2
                details["recommendations"].append("Move imports to the top of the file")
        
        # Check for consistent indentation
        indentations = []
        for line in non_empty_lines:
            if line.startswith(' ') or line.startswith('\t'):
                leading_spaces = len(line) - len(line.lstrip(' \t'))
                indentations.append(leading_spaces)
        
        if indentations:
            # Check for consistent indentation
            unique_indentations = set(indentations)
            if len(unique_indentations) > 4:  # Too many different indentation levels
                structure_score -= 0.3
                details["recommendations"].append("Use consistent indentation")
        
        # Check line length
        long_lines = [line for line in lines if len(line) > 100]
        if long_lines:
            structure_score -= min(0.2, len(long_lines) * 0.05)
            details["recommendations"].append("Consider breaking long lines")
        
        details["structure_analysis"] = {
            "import_organization": "good" if not import_lines or min(import_lines) <= len(lines) * 0.2 else "poor",
            "indentation_consistency": len(set(indentations)) if indentations else 0,
            "long_lines_count": len(long_lines),
            "structure_score": max(0.0, structure_score)
        }
        
        return max(0.0, structure_score)
    
    def get_evaluation_methods(self) -> List[EvaluationMethod]:
        """Return supported evaluation methods."""
        return [EvaluationMethod.STATIC_ANALYSIS]


class StaticEthicalAnalyzer(TrustworthinessEvaluator):
    """
    Static analysis for ethical considerations in code.
    """
    
    def __init__(self):
        super().__init__(TrustworthinessCategory.ETHICAL)
        self.ethical_patterns = self._setup_ethical_patterns()
    
    def _setup_ethical_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Setup patterns for detecting ethical issues."""
        return {
            "bias_indicators": {
                "patterns": [
                    r"\bgender\s*==\s*['\"]male['\"]",
                    r"\brace\s*==\s*['\"]white['\"]",
                    r"\bage\s*>\s*\d+",
                    r"\bif.*male.*:",
                    r"\bif.*female.*:"
                ],
                "severity": "medium",
                "description": "Potential bias in decision making"
            },
            "privacy_concerns": {
                "patterns": [
                    r"\bssn\b",
                    r"\bsocial.*security\b",
                    r"\bphone.*number\b",
                    r"\bemail.*address\b",
                    r"\bpersonal.*data\b"
                ],
                "severity": "medium",
                "description": "Handling of personal information"
            },
            "discriminatory_language": {
                "patterns": [
                    r"\bblacklist\b",
                    r"\bwhitelist\b",
                    r"\bmaster\b",
                    r"\bslave\b"
                ],
                "severity": "low",
                "description": "Potentially discriminatory terminology"
            }
        }
    
    async def evaluate(self, code_sample: CodeSample) -> EvaluationResult:
        """Evaluate ethical considerations through static analysis."""
        details = {
            "ethical_issues": [],
            "recommendations": [],
            "ethical_score_breakdown": {}
        }
        
        try:
            ethical_score = self._analyze_ethical_patterns(code_sample.source_code, details)
            confidence = 0.6  # Lower confidence for ethical analysis
            
        except Exception as e:
            details["analysis_error"] = str(e)
            ethical_score = 0.5  # Neutral score on error
            confidence = 0.3
        
        return EvaluationResult(
            category=self.category,
            method=EvaluationMethod.STATIC_ANALYSIS,
            score=ethical_score,
            confidence=confidence,
            details=details
        )
    
    def _analyze_ethical_patterns(self, code: str, details: Dict[str, Any]) -> float:
        """Analyze code for ethical patterns and issues."""
        ethical_score = 1.0
        issue_count = 0
        
        for issue_type, issue_info in self.ethical_patterns.items():
            for pattern in issue_info["patterns"]:
                matches = re.findall(pattern, code, re.IGNORECASE)
                if matches:
                    severity_weight = {"high": 0.4, "medium": 0.2, "low": 0.1}[issue_info["severity"]]
                    ethical_score -= len(matches) * severity_weight
                    issue_count += len(matches)
                    
                    details["ethical_issues"].append({
                        "type": issue_type,
                        "description": issue_info["description"],
                        "severity": issue_info["severity"],
                        "matches": len(matches),
                        "pattern": pattern
                    })
                    
                    # Add recommendations
                    if issue_type == "bias_indicators":
                        details["recommendations"].append(
                            "Review decision logic for potential bias. Consider fairness metrics."
                        )
                    elif issue_type == "privacy_concerns":
                        details["recommendations"].append(
                            "Ensure proper handling and protection of personal information."
                        )
                    elif issue_type == "discriminatory_language":
                        details["recommendations"].append(
                            "Consider using more inclusive terminology (e.g., 'allowlist'/'denylist')."
                        )
        
        details["ethical_score_breakdown"] = {
            "base_score": 1.0,
            "penalties_applied": 1.0 - ethical_score,
            "issue_count": issue_count,
            "final_score": max(0.0, ethical_score)
        }
        
        return max(0.0, ethical_score)
    
    def get_evaluation_methods(self) -> List[EvaluationMethod]:
        """Return supported evaluation methods."""
        return [EvaluationMethod.STATIC_ANALYSIS]
