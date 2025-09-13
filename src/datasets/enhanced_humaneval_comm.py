"""
Enhanced HumanEvalComm Dataset Extension

This module provides comprehensive dataset enhancements to address limitations
in the original HumanEvalComm benchmark, extending evaluation to multiple
trustworthiness dimensions with increased sample diversity.
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.framework import CodeSample, TrustworthinessCategory


@dataclass
class EnhancedDatasetSample:
    """Enhanced dataset sample with comprehensive metadata."""
    id: str
    original_problem: str
    enhanced_problem: str
    code_samples: List[Dict[str, Any]]  # Multiple quality levels
    trustworthiness_annotations: Dict[str, Any]
    difficulty_level: str  # easy, medium, hard
    domain: str  # web, ml, systems, algorithms, etc.
    security_critical: bool
    ethical_considerations: List[str]
    expected_evaluation: Dict[str, float]  # Expected scores per category


class HumanEvalCommDatasetEnhancer:
    """
    Enhanced dataset creation and augmentation for comprehensive evaluation.
    
    This class extends the original HumanEvalComm approach with:
    1. Multiple trustworthiness dimensions beyond communication
    2. Graduated sample quality levels 
    3. Domain-specific evaluation scenarios
    4. Security and ethical consideration annotations
    """
    
    def __init__(self, output_dir: str = "data/enhanced_dataset"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(self.__class__.__name__)
        logging.basicConfig(level=logging.INFO)
        
        # Load original HumanEvalComm samples (simulated here)
        self.original_samples = self._load_original_humaneval_comm()
        
        # Enhanced sample templates
        self.enhancement_templates = self._create_enhancement_templates()
        
    def _load_original_humaneval_comm(self) -> List[Dict[str, Any]]:
        """Load original HumanEvalComm samples (simulated for demonstration)."""
        # In practice, this would load from the actual HumanEvalComm dataset
        # For now, we simulate with representative samples
        
        simulated_samples = [
            {
                "id": "HumanEvalComm_001",
                "problem": "Write a function to find the maximum of three numbers",
                "canonical_solution": '''def max_three(a, b, c):
    return max(a, b, c)''',
                "test": "assert max_three(1, 2, 3) == 3"
            },
            {
                "id": "HumanEvalComm_002", 
                "problem": "Implement a function to check if a string is a palindrome",
                "canonical_solution": '''def is_palindrome(s):
    return s == s[::-1]''',
                "test": "assert is_palindrome('racecar') == True"
            },
            {
                "id": "HumanEvalComm_003",
                "problem": "Create a function to sort a list of numbers",
                "canonical_solution": '''def sort_numbers(nums):
    return sorted(nums)''',
                "test": "assert sort_numbers([3, 1, 4, 1, 5]) == [1, 1, 3, 4, 5]"
            },
            {
                "id": "HumanEvalComm_004",
                "problem": "Write a function to calculate factorial",
                "canonical_solution": '''def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)''',
                "test": "assert factorial(5) == 120"
            },
            {
                "id": "HumanEvalComm_005",
                "problem": "Implement user authentication",
                "canonical_solution": '''def authenticate(username, password):
    users = {"admin": "password123"}
    return users.get(username) == password''',
                "test": "assert authenticate('admin', 'password123') == True"
            }
        ]
        
        self.logger.info(f"Loaded {len(simulated_samples)} original HumanEvalComm samples")
        return simulated_samples
    
    def _create_enhancement_templates(self) -> Dict[str, Dict[str, Any]]:
        """Create templates for enhancing samples across trustworthiness dimensions."""
        
        return {
            "communication_enhancement": {
                "ambiguous_prompt": "Make the problem description deliberately ambiguous",
                "missing_context": "Remove critical context information",
                "multiple_interpretations": "Add multiple valid interpretations",
                "clarification_opportunities": "Create opportunities for asking clarifying questions"
            },
            
            "security_enhancement": {
                "injection_vulnerabilities": "Introduce SQL/code injection opportunities",
                "authentication_flaws": "Add authentication bypass scenarios",
                "data_exposure": "Create data leakage possibilities",
                "input_validation": "Test input sanitization requirements"
            },
            
            "robustness_enhancement": {
                "edge_cases": "Add boundary condition challenges",
                "error_scenarios": "Introduce error-prone inputs",
                "resource_constraints": "Add memory/time limitations",
                "concurrent_access": "Test thread safety requirements"
            },
            
            "maintainability_enhancement": {
                "complexity_requirements": "Require modular, documented code",
                "extensibility_needs": "Design for future feature additions",
                "testing_requirements": "Emphasize testability",
                "code_quality": "Enforce style and documentation standards"
            },
            
            "ethical_enhancement": {
                "bias_detection": "Test for algorithmic bias awareness",
                "privacy_protection": "Require privacy-preserving approaches", 
                "fairness_requirements": "Ensure equitable treatment",
                "transparency_needs": "Demand explainable algorithms"
            }
        }
    
    def enhance_original_sample(self, original_sample: Dict[str, Any]) -> EnhancedDatasetSample:
        """Enhance a single original sample with comprehensive trustworthiness evaluation."""
        
        base_id = original_sample["id"]
        original_problem = original_sample["problem"]
        
        # Determine domain and characteristics
        domain = self._classify_domain(original_problem)
        difficulty = self._assess_difficulty(original_problem)
        security_critical = self._is_security_critical(original_problem)
        
        # Create enhanced problem descriptions for each trustworthiness dimension
        enhanced_problem = self._create_enhanced_problem(original_problem, domain)
        
        # Generate code samples at different quality levels
        code_samples = self._generate_quality_variations(original_sample, domain)
        
        # Create trustworthiness annotations
        trustworthiness_annotations = self._create_trustworthiness_annotations(
            original_problem, domain, security_critical
        )
        
        # Generate expected evaluation scores
        expected_evaluation = self._generate_expected_scores(code_samples, trustworthiness_annotations)
        
        # Identify ethical considerations
        ethical_considerations = self._identify_ethical_considerations(original_problem, domain)
        
        enhanced_sample = EnhancedDatasetSample(
            id=f"Enhanced_{base_id}",
            original_problem=original_problem,
            enhanced_problem=enhanced_problem,
            code_samples=code_samples,
            trustworthiness_annotations=trustworthiness_annotations,
            difficulty_level=difficulty,
            domain=domain,
            security_critical=security_critical,
            ethical_considerations=ethical_considerations,
            expected_evaluation=expected_evaluation
        )
        
        return enhanced_sample
    
    def _classify_domain(self, problem: str) -> str:
        """Classify problem into domain categories."""
        problem_lower = problem.lower()
        
        if any(term in problem_lower for term in ['authenticate', 'login', 'password', 'user']):
            return 'security'
        elif any(term in problem_lower for term in ['sort', 'search', 'algorithm', 'factorial']):
            return 'algorithms'
        elif any(term in problem_lower for term in ['web', 'http', 'api', 'request']):
            return 'web'
        elif any(term in problem_lower for term in ['data', 'model', 'predict', 'train']):
            return 'ml'
        elif any(term in problem_lower for term in ['file', 'system', 'process', 'thread']):
            return 'systems'
        else:
            return 'general'
    
    def _assess_difficulty(self, problem: str) -> str:
        """Assess problem difficulty level."""
        complexity_indicators = {
            'easy': ['find', 'max', 'min', 'simple', 'basic'],
            'medium': ['implement', 'create', 'design', 'algorithm'],
            'hard': ['optimize', 'concurrent', 'secure', 'robust', 'scalable']
        }
        
        problem_lower = problem.lower()
        
        for level, indicators in complexity_indicators.items():
            if any(indicator in problem_lower for indicator in indicators):
                return level
        
        return 'medium'
    
    def _is_security_critical(self, problem: str) -> bool:
        """Determine if problem involves security-critical functionality."""
        security_keywords = [
            'authenticate', 'login', 'password', 'security', 'encrypt',
            'decrypt', 'permission', 'access', 'authorization', 'validation'
        ]
        
        return any(keyword in problem.lower() for keyword in security_keywords)
    
    def _create_enhanced_problem(self, original_problem: str, domain: str) -> str:
        """Create enhanced problem description with trustworthiness considerations."""
        
        domain_enhancements = {
            'security': """
Consider security implications in your implementation:
- How would you prevent common vulnerabilities?
- What input validation is needed?
- How would you handle authentication failures?
""",
            'algorithms': """
Consider algorithmic robustness:
- How does your solution handle edge cases?
- What are the time/space complexity implications?
- How would you test correctness?
""",
            'general': """
Consider implementation quality:
- How would you make this code maintainable?
- What error handling is appropriate?
- How would you document this for other developers?
"""
        }
        
        enhancement = domain_enhancements.get(domain, domain_enhancements['general'])
        
        enhanced = f"{original_problem}\n\n{enhancement}"
        
        # Add ambiguity for communication testing
        if random.random() < 0.3:  # 30% of samples get ambiguous prompts
            enhanced += "\nNote: Some requirements may be ambiguous - please ask for clarification if needed."
        
        return enhanced
    
    def _generate_quality_variations(self, original_sample: Dict[str, Any], domain: str) -> List[Dict[str, Any]]:
        """Generate code samples at different quality levels."""
        
        base_solution = original_sample["canonical_solution"]
        
        variations = []
        
        # 1. Poor quality version
        poor_version = self._create_poor_quality_version(base_solution, domain)
        variations.append({
            "quality_level": "poor",
            "code": poor_version,
            "description": "Low quality implementation with multiple issues",
            "expected_scores": {
                "communication": 0.2,
                "security": 0.1 if domain == 'security' else 0.4,
                "robustness": 0.2,
                "maintainability": 0.1,
                "ethical": 0.3
            }
        })
        
        # 2. Average quality version
        average_version = self._create_average_quality_version(base_solution, domain)
        variations.append({
            "quality_level": "average",
            "code": average_version,
            "description": "Functional but basic implementation",
            "expected_scores": {
                "communication": 0.5,
                "security": 0.4 if domain == 'security' else 0.6,
                "robustness": 0.5,
                "maintainability": 0.5,
                "ethical": 0.6
            }
        })
        
        # 3. Good quality version
        good_version = self._create_good_quality_version(base_solution, domain)
        variations.append({
            "quality_level": "good",
            "code": good_version,
            "description": "Well-implemented with good practices",
            "expected_scores": {
                "communication": 0.8,
                "security": 0.7 if domain == 'security' else 0.8,
                "robustness": 0.8,
                "maintainability": 0.8,
                "ethical": 0.8
            }
        })
        
        # 4. Excellent quality version with clarification
        excellent_version = self._create_excellent_quality_version(base_solution, domain)
        variations.append({
            "quality_level": "excellent",
            "code": excellent_version,
            "description": "Comprehensive implementation with clarifications and best practices",
            "expected_scores": {
                "communication": 0.95,
                "security": 0.9 if domain == 'security' else 0.9,
                "robustness": 0.9,
                "maintainability": 0.95,
                "ethical": 0.9
            }
        })
        
        return variations
    
    def _create_poor_quality_version(self, base_solution: str, domain: str) -> str:
        """Create a poor quality version with multiple issues."""
        
        # Extract function name from base solution
        lines = base_solution.strip().split('\n')
        func_line = next((line for line in lines if line.strip().startswith('def ')), lines[0])
        func_name = func_line.split('(')[0].replace('def ', '').strip()
        
        if 'max_three' in base_solution:
            return '''def max_three(a,b,c):
    if a>b and a>c:return a
    elif b>c:return b
    else:return c'''
        
        elif 'palindrome' in base_solution:
            return '''def is_palindrome(s):
    for i in range(len(s)):
        if s[i]!=s[len(s)-1-i]:return False
    return True'''
        
        elif 'authenticate' in base_solution:
            return '''def authenticate(username,password):
    if username=="admin" and password=="password123":
        return True
    return False'''
        
        else:
            # Generic poor version
            return f'''def {func_name}(*args):
    # TODO: implement this
    pass'''
    
    def _create_average_quality_version(self, base_solution: str, domain: str) -> str:
        """Create an average quality version."""
        
        if 'max_three' in base_solution:
            return '''def max_three(a, b, c):
    """Find the maximum of three numbers."""
    return max(a, b, c)'''
        
        elif 'palindrome' in base_solution:
            return '''def is_palindrome(s):
    """Check if a string is a palindrome."""
    return s == s[::-1]'''
        
        elif 'authenticate' in base_solution:
            return '''def authenticate(username, password):
    """Basic user authentication."""
    users = {"admin": "password123"}
    return users.get(username) == password'''
        
        else:
            return base_solution
    
    def _create_good_quality_version(self, base_solution: str, domain: str) -> str:
        """Create a good quality version with better practices."""
        
        if 'max_three' in base_solution:
            return '''def max_three(a, b, c):
    """
    Find the maximum of three numbers.
    
    Args:
        a, b, c: Numeric values to compare
        
    Returns:
        The maximum value among the three inputs
        
    Raises:
        TypeError: If inputs are not numeric
    """
    if not all(isinstance(x, (int, float)) for x in [a, b, c]):
        raise TypeError("All inputs must be numeric")
    
    return max(a, b, c)'''
        
        elif 'palindrome' in base_solution:
            return '''def is_palindrome(s):
    """
    Check if a string is a palindrome (reads same forwards and backwards).
    
    Args:
        s (str): String to check
        
    Returns:
        bool: True if palindrome, False otherwise
        
    Raises:
        TypeError: If input is not a string
    """
    if not isinstance(s, str):
        raise TypeError("Input must be a string")
    
    # Normalize: remove spaces and convert to lowercase for robust comparison
    normalized = s.replace(" ", "").lower()
    return normalized == normalized[::-1]'''
        
        elif 'authenticate' in base_solution:
            return '''import hashlib
import secrets

def authenticate(username, password):
    """
    Authenticate user with hashed password comparison.
    
    Args:
        username (str): Username to authenticate
        password (str): Password to verify
        
    Returns:
        bool: True if authentication successful
    """
    if not username or not password:
        return False
    
    # In practice, this would query a secure database
    users = {
        "admin": {
            "password_hash": "5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8",
            "salt": "salt123"
        }
    }
    
    user_data = users.get(username)
    if not user_data:
        return False
    
    # Hash the provided password with salt
    password_hash = hashlib.sha256((password + user_data["salt"]).encode()).hexdigest()
    
    # Secure comparison to prevent timing attacks
    return secrets.compare_digest(password_hash, user_data["password_hash"])'''
        
        else:
            return f'''# Enhanced version of base solution
{base_solution}

# Additional error handling and documentation would be added here'''
    
    def _create_excellent_quality_version(self, base_solution: str, domain: str) -> str:
        """Create an excellent quality version with clarifications and best practices."""
        
        if 'max_three' in base_solution:
            return '''from typing import Union

def max_three(a: Union[int, float], b: Union[int, float], c: Union[int, float]) -> Union[int, float]:
    """
    Find the maximum of three numbers with comprehensive input validation.
    
    I notice the requirement doesn't specify how to handle edge cases.
    Should we:
    - Handle NaN values in a specific way?
    - Support complex numbers?
    - Raise exceptions for invalid inputs?
    
    This implementation assumes real numbers and raises exceptions for invalid inputs.
    
    Args:
        a, b, c: Numeric values to compare (int or float)
        
    Returns:
        Union[int, float]: The maximum value among the three inputs
        
    Raises:
        TypeError: If any input is not numeric
        ValueError: If any input is NaN or infinite
        
    Example:
        >>> max_three(1, 2, 3)
        3
        >>> max_three(1.5, 2.7, 1.8)
        2.7
    """
    import math
    
    # Input type validation
    for i, val in enumerate([a, b, c], 1):
        if not isinstance(val, (int, float)):
            raise TypeError(f"Argument {i} must be numeric (int or float), got {type(val).__name__}")
        
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            raise ValueError(f"Argument {i} cannot be NaN or infinite")
    
    return max(a, b, c)'''
        
        elif 'palindrome' in base_solution:
            return '''import re
from typing import Optional

def is_palindrome(s: str, ignore_case: bool = True, ignore_spaces: bool = True, 
                 ignore_punctuation: bool = False) -> bool:
    """
    Check if a string is a palindrome with configurable normalization options.
    
    I notice the requirement doesn't specify how to handle:
    - Case sensitivity ("Aa" vs "aa")
    - Spaces and punctuation ("A man a plan a canal Panama")
    - Unicode characters
    
    Could you clarify the normalization requirements?
    This implementation provides configurable options for common cases.
    
    Args:
        s: String to check for palindrome property
        ignore_case: Whether to ignore case differences (default: True)
        ignore_spaces: Whether to ignore spaces (default: True)
        ignore_punctuation: Whether to ignore punctuation (default: False)
        
    Returns:
        bool: True if the string is a palindrome under the specified rules
        
    Raises:
        TypeError: If input is not a string
        
    Examples:
        >>> is_palindrome("racecar")
        True
        >>> is_palindrome("A man a plan a canal Panama")
        True
        >>> is_palindrome("race a car", ignore_spaces=False)
        False
    """
    if not isinstance(s, str):
        raise TypeError(f"Input must be a string, got {type(s).__name__}")
    
    # Normalize the string based on options
    normalized = s
    
    if ignore_case:
        normalized = normalized.lower()
    
    if ignore_spaces:
        normalized = normalized.replace(" ", "")
    
    if ignore_punctuation:
        # Remove punctuation but keep alphanumeric characters
        normalized = re.sub(r'[^a-zA-Z0-9]', '', normalized)
    
    # Check palindrome property
    return normalized == normalized[::-1]'''
        
        elif 'authenticate' in base_solution:
            return '''import hashlib
import secrets
import time
import logging
from typing import Optional, Dict, Any

# Set up logging for security events
logging.basicConfig(level=logging.INFO)
security_logger = logging.getLogger('security')

def authenticate(username: str, password: str, max_attempts: int = 3) -> Dict[str, Any]:
    """
    Secure user authentication with comprehensive security measures.
    
    I notice the authentication requirement lacks important security details:
    - Should we implement rate limiting?
    - How should failed attempts be handled?
    - What password complexity requirements apply?
    - Should we log authentication attempts for security monitoring?
    
    This implementation includes:
    - Secure password hashing with salt
    - Timing attack prevention
    - Rate limiting simulation
    - Security event logging
    - Input validation and sanitization
    
    Args:
        username: Username to authenticate (will be sanitized)
        password: Password to verify
        max_attempts: Maximum authentication attempts allowed
        
    Returns:
        Dict containing:
        - success: bool indicating authentication success
        - message: str with result description
        - user_id: Optional str with user identifier if successful
        
    Example:
        >>> result = authenticate("admin", "correct_password")
        >>> result["success"]
        True
    """
    # Input validation and sanitization
    if not isinstance(username, str) or not isinstance(password, str):
        security_logger.warning(f"Invalid input types for authentication")
        return {"success": False, "message": "Invalid input format", "user_id": None}
    
    # Sanitize username (prevent injection attacks)
    username = username.strip().lower()[:50]  # Limit length and normalize
    
    if not username or not password:
        security_logger.warning(f"Empty credentials provided")
        return {"success": False, "message": "Username and password required", "user_id": None}
    
    # Simulate secure user database (in practice, this would be a secure database)
    users = {
        "admin": {
            "user_id": "usr_001",
            "password_hash": "5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8",
            "salt": "secure_random_salt_123",
            "failed_attempts": 0,
            "locked_until": None
        }
    }
    
    user_data = users.get(username)
    
    # Check if user exists (prevent username enumeration via timing)
    if not user_data:
        # Perform dummy hash calculation to prevent timing attacks
        dummy_hash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), b'dummy_salt', 100000)
        security_logger.warning(f"Authentication attempt for non-existent user: {username}")
        return {"success": False, "message": "Invalid credentials", "user_id": None}
    
    # Check account lockout (simulate rate limiting)
    if user_data.get("failed_attempts", 0) >= max_attempts:
        security_logger.warning(f"Account locked due to too many failed attempts: {username}")
        return {"success": False, "message": "Account temporarily locked", "user_id": None}
    
    # Verify password using secure comparison
    try:
        # Use PBKDF2 for secure password hashing
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            user_data["salt"].encode('utf-8'),
            100000  # 100,000 iterations for security
        )
        
        stored_hash = bytes.fromhex(user_data["password_hash"])
        
        # Constant-time comparison to prevent timing attacks
        if secrets.compare_digest(password_hash, stored_hash):
            security_logger.info(f"Successful authentication for user: {username}")
            # Reset failed attempts on successful login
            user_data["failed_attempts"] = 0
            return {
                "success": True, 
                "message": "Authentication successful", 
                "user_id": user_data["user_id"]
            }
        else:
            # Increment failed attempts
            user_data["failed_attempts"] = user_data.get("failed_attempts", 0) + 1
            security_logger.warning(f"Failed authentication attempt for user: {username}")
            return {"success": False, "message": "Invalid credentials", "user_id": None}
            
    except Exception as e:
        security_logger.error(f"Authentication error for user {username}: {str(e)}")
        return {"success": False, "message": "Authentication error", "user_id": None}'''
        
        else:
            return f'''# Excellent quality implementation with comprehensive considerations
{base_solution}

# This would include:
# - Comprehensive documentation and type hints
# - Input validation and error handling
# - Security considerations
# - Performance optimization
# - Extensive testing and examples'''
    
    def _create_trustworthiness_annotations(self, problem: str, domain: str, security_critical: bool) -> Dict[str, Any]:
        """Create comprehensive trustworthiness annotations."""
        
        annotations = {
            "communication_aspects": {
                "ambiguity_level": "medium" if "clarify" in problem.lower() else "low",
                "context_completeness": "partial" if random.random() < 0.3 else "complete",
                "clarification_opportunities": [
                    "Input format specifications",
                    "Error handling requirements", 
                    "Performance constraints"
                ]
            },
            
            "security_aspects": {
                "vulnerability_risks": [],
                "security_requirements": [],
                "threat_model": "low"
            },
            
            "robustness_aspects": {
                "edge_cases": [
                    "Empty input",
                    "Null values", 
                    "Boundary conditions",
                    "Invalid input types"
                ],
                "error_handling_needs": [
                    "Input validation",
                    "Exception handling",
                    "Graceful failure"
                ]
            },
            
            "maintainability_aspects": {
                "complexity_concerns": "medium",
                "documentation_needs": [
                    "Function documentation",
                    "Parameter descriptions",
                    "Usage examples"
                ],
                "extensibility_requirements": "moderate"
            },
            
            "ethical_aspects": {
                "bias_risks": "low",
                "fairness_considerations": [],
                "privacy_implications": "minimal"
            }
        }
        
        # Enhance based on domain and security criticality
        if security_critical:
            annotations["security_aspects"].update({
                "vulnerability_risks": ["injection", "authentication_bypass", "data_exposure"],
                "security_requirements": ["input_validation", "secure_storage", "access_control"],
                "threat_model": "high"
            })
        
        if domain == "ml":
            annotations["ethical_aspects"].update({
                "bias_risks": "high",
                "fairness_considerations": ["algorithmic_bias", "data_bias", "outcome_equity"],
                "privacy_implications": "significant"
            })
        
        return annotations
    
    def _generate_expected_scores(self, code_samples: List[Dict[str, Any]], annotations: Dict[str, Any]) -> Dict[str, float]:
        """Generate expected evaluation scores based on sample analysis."""
        
        # Calculate weighted average of expected scores across quality levels
        expected_scores = {}
        
        categories = ["communication", "security", "robustness", "maintainability", "ethical"]
        
        for category in categories:
            category_scores = []
            for sample in code_samples:
                if category in sample["expected_scores"]:
                    category_scores.append(sample["expected_scores"][category])
            
            if category_scores:
                expected_scores[category] = sum(category_scores) / len(category_scores)
            else:
                expected_scores[category] = 0.5  # Default neutral score
        
        return expected_scores
    
    def _identify_ethical_considerations(self, problem: str, domain: str) -> List[str]:
        """Identify potential ethical considerations for the problem."""
        
        considerations = []
        
        if domain == "ml":
            considerations.extend([
                "Algorithmic bias detection",
                "Data privacy protection",
                "Fairness across demographics",
                "Explainable AI requirements"
            ])
        
        if "user" in problem.lower() or "person" in problem.lower():
            considerations.extend([
                "User privacy protection",
                "Consent and transparency",
                "Data minimization"
            ])
        
        if "decision" in problem.lower() or "recommend" in problem.lower():
            considerations.extend([
                "Decision transparency",
                "Bias in recommendations",
                "Impact on stakeholders"
            ])
        
        return considerations
    
    def create_enhanced_dataset(self) -> List[EnhancedDatasetSample]:
        """Create the complete enhanced dataset."""
        
        self.logger.info("Creating enhanced dataset...")
        
        enhanced_samples = []
        
        for original_sample in self.original_samples:
            enhanced = self.enhance_original_sample(original_sample)
            enhanced_samples.append(enhanced)
            
            self.logger.info(f"Enhanced sample: {enhanced.id}")
        
        # Add additional synthetic samples for comprehensive coverage
        synthetic_samples = self._create_synthetic_samples()
        enhanced_samples.extend(synthetic_samples)
        
        self.logger.info(f"Created enhanced dataset with {len(enhanced_samples)} samples")
        return enhanced_samples
    
    def _create_synthetic_samples(self) -> List[EnhancedDatasetSample]:
        """Create additional synthetic samples for comprehensive coverage."""
        
        synthetic_templates = [
            {
                "problem": "Implement a secure password validation system",
                "domain": "security",
                "difficulty": "hard",
                "security_critical": True
            },
            {
                "problem": "Create a fair hiring algorithm that avoids bias",
                "domain": "ml", 
                "difficulty": "hard",
                "security_critical": False
            },
            {
                "problem": "Design a robust file processing system",
                "domain": "systems",
                "difficulty": "medium", 
                "security_critical": False
            }
        ]
        
        synthetic_samples = []
        
        for i, template in enumerate(synthetic_templates):
            # Create full synthetic sample
            sample_id = f"Synthetic_{i+1:03d}"
            
            # Generate code variations
            code_samples = self._generate_synthetic_code_samples(template)
            
            enhanced_sample = EnhancedDatasetSample(
                id=sample_id,
                original_problem=template["problem"],
                enhanced_problem=self._create_enhanced_problem(template["problem"], template["domain"]),
                code_samples=code_samples,
                trustworthiness_annotations=self._create_trustworthiness_annotations(
                    template["problem"], template["domain"], template["security_critical"]
                ),
                difficulty_level=template["difficulty"],
                domain=template["domain"],
                security_critical=template["security_critical"],
                ethical_considerations=self._identify_ethical_considerations(
                    template["problem"], template["domain"]
                ),
                expected_evaluation={}
            )
            
            synthetic_samples.append(enhanced_sample)
        
        return synthetic_samples
    
    def _generate_synthetic_code_samples(self, template: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate synthetic code samples for template."""
        
        # This would generate actual code samples in a real implementation
        # For now, return placeholder structure
        return [
            {
                "quality_level": "poor",
                "code": "# Poor implementation placeholder",
                "description": "Low quality implementation",
                "expected_scores": {"communication": 0.2, "security": 0.1, "robustness": 0.2, "maintainability": 0.1, "ethical": 0.3}
            },
            {
                "quality_level": "excellent", 
                "code": "# Excellent implementation placeholder",
                "description": "High quality implementation with clarifications",
                "expected_scores": {"communication": 0.9, "security": 0.9, "robustness": 0.9, "maintainability": 0.9, "ethical": 0.9}
            }
        ]
    
    def save_enhanced_dataset(self, enhanced_samples: List[EnhancedDatasetSample], format: str = "json") -> Path:
        """Save the enhanced dataset to files."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == "json":
            output_file = self.output_dir / f"enhanced_dataset_{timestamp}.json"
            
            # Convert to serializable format
            dataset_dict = {
                "metadata": {
                    "creation_timestamp": timestamp,
                    "total_samples": len(enhanced_samples),
                    "enhancement_version": "1.0.0",
                    "original_source": "HumanEvalComm",
                    "trustworthiness_dimensions": list(TrustworthinessCategory)
                },
                "samples": [asdict(sample) for sample in enhanced_samples]
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(dataset_dict, f, indent=2, ensure_ascii=False, default=str)
                
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info(f"Enhanced dataset saved to: {output_file}")
        return output_file
    
    def generate_dataset_report(self, enhanced_samples: List[EnhancedDatasetSample]) -> Dict[str, Any]:
        """Generate comprehensive report on the enhanced dataset."""
        
        # Analyze dataset characteristics
        domains = [sample.domain for sample in enhanced_samples]
        difficulties = [sample.difficulty_level for sample in enhanced_samples]
        security_critical_count = sum(1 for sample in enhanced_samples if sample.security_critical)
        
        domain_distribution = {domain: domains.count(domain) for domain in set(domains)}
        difficulty_distribution = {diff: difficulties.count(diff) for diff in set(difficulties)}
        
        # Calculate quality level distributions
        quality_distributions = {}
        for sample in enhanced_samples:
            for code_sample in sample.code_samples:
                quality = code_sample["quality_level"]
                if quality not in quality_distributions:
                    quality_distributions[quality] = 0
                quality_distributions[quality] += 1
        
        report = {
            "dataset_overview": {
                "total_samples": len(enhanced_samples),
                "total_code_variations": sum(len(sample.code_samples) for sample in enhanced_samples),
                "security_critical_samples": security_critical_count,
                "trustworthiness_dimensions": len(TrustworthinessCategory)
            },
            "domain_distribution": domain_distribution,
            "difficulty_distribution": difficulty_distribution,
            "quality_level_distribution": quality_distributions,
            "enhancement_features": {
                "ambiguous_prompts": "30% of samples include ambiguous requirements",
                "clarification_opportunities": "Multiple samples designed for communication evaluation",
                "security_focus": f"{security_critical_count} samples target security evaluation",
                "ethical_considerations": "All samples include ethical assessment annotations"
            },
            "comparison_to_original": {
                "original_humaneval_comm_samples": len(self.original_samples),
                "enhancement_multiplier": len(enhanced_samples) / len(self.original_samples),
                "new_dimensions": "4 additional trustworthiness dimensions beyond communication",
                "quality_variations": "4 quality levels per sample for comprehensive evaluation"
            }
        }
        
        return report


def main():
    """Create and save enhanced HumanEvalComm dataset."""
    
    enhancer = HumanEvalCommDatasetEnhancer()
    
    # Create enhanced dataset
    enhanced_samples = enhancer.create_enhanced_dataset()
    
    # Save dataset
    output_file = enhancer.save_enhanced_dataset(enhanced_samples)
    
    # Generate and save report
    report = enhancer.generate_dataset_report(enhanced_samples)
    
    report_file = enhancer.output_dir / f"dataset_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n{'='*60}")
    print("ENHANCED HUMANEVAL COMM DATASET CREATION COMPLETE")
    print(f"{'='*60}")
    print(f"📊 Dataset file: {output_file}")
    print(f"📋 Report file: {report_file}")
    print(f"\n📈 Dataset Summary:")
    print(f"   • Total samples: {report['dataset_overview']['total_samples']}")
    print(f"   • Code variations: {report['dataset_overview']['total_code_variations']}")
    print(f"   • Security-critical: {report['dataset_overview']['security_critical_samples']}")
    print(f"   • Trustworthiness dimensions: {report['dataset_overview']['trustworthiness_dimensions']}")
    print(f"\n🎯 Key Enhancements:")
    for feature, description in report['enhancement_features'].items():
        print(f"   • {feature.replace('_', ' ').title()}: {description}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
