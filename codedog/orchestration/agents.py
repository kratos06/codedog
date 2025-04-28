"""
Specialized agents for different aspects of code review.
"""

import logging
from typing import Dict, Any, Optional

from codedog.orchestration.base_agent import BaseReviewAgent

logger = logging.getLogger(__name__)

class SecurityReviewAgent(BaseReviewAgent):
    """
    Specialized agent for security code review.
    
    This agent focuses on identifying security vulnerabilities,
    potential exploits, and secure coding practices.
    """
    
    def __init__(self, model):
        """Initialize the security review agent."""
        system_prompt = """You are a specialized security code review agent within an orchestrated code review system.
Your primary focus is identifying security vulnerabilities, potential exploits, and secure coding practices.

Evaluate the code specifically for:
1. Input validation and sanitization
2. Authentication and authorization issues
3. Sensitive data exposure
4. Injection vulnerabilities (SQL, command, etc.)
5. Cross-site scripting (XSS) and cross-site request forgery (CSRF)
6. Insecure cryptographic implementations
7. Hardcoded credentials or secrets
8. Proper error handling that doesn't leak sensitive information
9. Race conditions and concurrency issues
10. Compliance with security standards (OWASP, etc.)

For each issue found, provide:
1. A clear description of the vulnerability
2. The potential impact and risk level
3. Specific code location(s) where the issue exists
4. Concrete recommendations for remediation with code examples

Your output should include:
1. A security score (1-10) for the code
2. Detailed analysis of security issues found
3. Prioritized recommendations for improvement
4. Estimated hours for an experienced developer to fix the issues

Format your response with clear sections and markdown formatting for readability.
"""
        super().__init__(model, system_prompt)
    
    async def review(self, code_diff: Dict[str, Any], context: Dict[str, Any], priority: bool = False) -> Dict[str, Any]:
        """Perform a security-focused code review."""
        logger.info("Performing security review")
        
        # Extract information from code_diff
        file_path = code_diff.get("file_path", "")
        language = code_diff.get("language", "")
        content = code_diff.get("content", "")
        
        # Create prompt for security review
        prompt = f"""# Security Code Review Request

## File Information
- **File Path**: {file_path}
- **Language**: {language}
- **Priority**: {"High" if priority else "Normal"}

## Code to Review
```{language}
{content}
```

## Context
- **File Type**: {context.get('file_type', 'Unknown')}
- **Complexity**: {context.get('estimated_complexity', 'Unknown')}

## Instructions
Please conduct a thorough security review of this code, focusing on:
1. Identifying any security vulnerabilities
2. Assessing the risk level of each issue
3. Providing specific recommendations for remediation
4. Estimating the work hours required to fix the issues

## Response Format
Please structure your response with the following sections:

### SUMMARY
Brief overview of security findings

### VULNERABILITIES
Detailed analysis of each security issue found

### RECOMMENDATIONS
Prioritized list of security improvements

### SCORES:
- Readability: [score] /10
- Efficiency: [score] /10
- Security: [score] /10
- Structure: [score] /10
- Error Handling: [score] /10
- Documentation: [score] /10
- Code Style: [score] /10
- Overall Score: [score] /10

### ESTIMATED HOURS: [hours]
"""
        
        # Generate review
        review_text = await self._generate_review(prompt, "security_review")
        
        # Parse scores and estimated hours
        scores = self._parse_scores(review_text)
        estimated_hours = self._parse_estimated_hours(review_text)
        
        # Return review results
        return {
            "scores": scores,
            "feedback": review_text,
            "estimated_hours": estimated_hours
        }


class PerformanceReviewAgent(BaseReviewAgent):
    """
    Specialized agent for performance code review.
    
    This agent focuses on identifying performance issues,
    optimization opportunities, and efficient coding practices.
    """
    
    def __init__(self, model):
        """Initialize the performance review agent."""
        system_prompt = """You are a specialized performance code review agent within an orchestrated code review system.
Your primary focus is identifying performance issues, optimization opportunities, and efficient coding practices.

Evaluate the code specifically for:
1. Algorithmic efficiency and complexity (O notation)
2. Resource utilization (CPU, memory, network, disk)
3. Caching opportunities and implementations
4. Database query optimization
5. Concurrency and parallelism
6. Memory leaks and garbage collection
7. Unnecessary computations or operations
8. Efficient data structures and algorithms
9. Performance bottlenecks
10. Scalability concerns

For each issue found, provide:
1. A clear description of the performance issue
2. The potential impact on system performance
3. Specific code location(s) where the issue exists
4. Concrete recommendations for optimization with code examples

Your output should include:
1. An efficiency score (1-10) for the code
2. Detailed analysis of performance issues found
3. Prioritized recommendations for improvement
4. Estimated hours for an experienced developer to implement optimizations

Format your response with clear sections and markdown formatting for readability.
"""
        super().__init__(model, system_prompt)
    
    async def review(self, code_diff: Dict[str, Any], context: Dict[str, Any], priority: bool = False) -> Dict[str, Any]:
        """Perform a performance-focused code review."""
        logger.info("Performing performance review")
        
        # Extract information from code_diff
        file_path = code_diff.get("file_path", "")
        language = code_diff.get("language", "")
        content = code_diff.get("content", "")
        
        # Create prompt for performance review
        prompt = f"""# Performance Code Review Request

## File Information
- **File Path**: {file_path}
- **Language**: {language}
- **Priority**: {"High" if priority else "Normal"}

## Code to Review
```{language}
{content}
```

## Context
- **File Type**: {context.get('file_type', 'Unknown')}
- **Complexity**: {context.get('estimated_complexity', 'Unknown')}

## Instructions
Please conduct a thorough performance review of this code, focusing on:
1. Identifying any performance issues or bottlenecks
2. Assessing the algorithmic complexity
3. Providing specific optimization recommendations
4. Estimating the work hours required to implement optimizations

## Response Format
Please structure your response with the following sections:

### SUMMARY
Brief overview of performance findings

### PERFORMANCE ISSUES
Detailed analysis of each performance issue found

### OPTIMIZATION RECOMMENDATIONS
Prioritized list of performance improvements

### SCORES:
- Readability: [score] /10
- Efficiency: [score] /10
- Security: [score] /10
- Structure: [score] /10
- Error Handling: [score] /10
- Documentation: [score] /10
- Code Style: [score] /10
- Overall Score: [score] /10

### ESTIMATED HOURS: [hours]
"""
        
        # Generate review
        review_text = await self._generate_review(prompt, "performance_review")
        
        # Parse scores and estimated hours
        scores = self._parse_scores(review_text)
        estimated_hours = self._parse_estimated_hours(review_text)
        
        # Return review results
        return {
            "scores": scores,
            "feedback": review_text,
            "estimated_hours": estimated_hours
        }


class ReadabilityReviewAgent(BaseReviewAgent):
    """
    Specialized agent for readability code review.
    
    This agent focuses on code readability, naming conventions,
    formatting, and overall code clarity.
    """
    
    def __init__(self, model):
        """Initialize the readability review agent."""
        system_prompt = """You are a specialized readability code review agent within an orchestrated code review system.
Your primary focus is evaluating code readability, naming conventions, formatting, and overall code clarity.

Evaluate the code specifically for:
1. Clear and descriptive variable, function, and class names
2. Consistent formatting and indentation
3. Appropriate use of whitespace
4. Code organization and logical flow
5. Adherence to language-specific style guides
6. Avoidance of overly complex expressions
7. Appropriate use of comments
8. Function and method length
9. Nesting depth
10. Overall code clarity and maintainability

For each issue found, provide:
1. A clear description of the readability issue
2. The impact on code maintainability
3. Specific code location(s) where the issue exists
4. Concrete recommendations for improvement with code examples

Your output should include:
1. A readability score (1-10) for the code
2. Detailed analysis of readability issues found
3. Prioritized recommendations for improvement
4. Estimated hours for an experienced developer to implement improvements

Format your response with clear sections and markdown formatting for readability.
"""
        super().__init__(model, system_prompt)
    
    async def review(self, code_diff: Dict[str, Any], context: Dict[str, Any], priority: bool = False) -> Dict[str, Any]:
        """Perform a readability-focused code review."""
        logger.info("Performing readability review")
        
        # Extract information from code_diff
        file_path = code_diff.get("file_path", "")
        language = code_diff.get("language", "")
        content = code_diff.get("content", "")
        
        # Create prompt for readability review
        prompt = f"""# Readability Code Review Request

## File Information
- **File Path**: {file_path}
- **Language**: {language}
- **Priority**: {"High" if priority else "Normal"}

## Code to Review
```{language}
{content}
```

## Context
- **File Type**: {context.get('file_type', 'Unknown')}
- **Complexity**: {context.get('estimated_complexity', 'Unknown')}

## Instructions
Please conduct a thorough readability review of this code, focusing on:
1. Identifying any readability issues
2. Assessing naming conventions and code clarity
3. Providing specific recommendations for improvement
4. Estimating the work hours required to implement improvements

## Response Format
Please structure your response with the following sections:

### SUMMARY
Brief overview of readability findings

### READABILITY ISSUES
Detailed analysis of each readability issue found

### IMPROVEMENT RECOMMENDATIONS
Prioritized list of readability improvements

### SCORES:
- Readability: [score] /10
- Efficiency: [score] /10
- Security: [score] /10
- Structure: [score] /10
- Error Handling: [score] /10
- Documentation: [score] /10
- Code Style: [score] /10
- Overall Score: [score] /10

### ESTIMATED HOURS: [hours]
"""
        
        # Generate review
        review_text = await self._generate_review(prompt, "readability_review")
        
        # Parse scores and estimated hours
        scores = self._parse_scores(review_text)
        estimated_hours = self._parse_estimated_hours(review_text)
        
        # Return review results
        return {
            "scores": scores,
            "feedback": review_text,
            "estimated_hours": estimated_hours
        }


class ArchitectureReviewAgent(BaseReviewAgent):
    """
    Specialized agent for architecture code review.
    
    This agent focuses on code structure, design patterns,
    architectural principles, and overall code organization.
    """
    
    def __init__(self, model):
        """Initialize the architecture review agent."""
        system_prompt = """You are a specialized architecture code review agent within an orchestrated code review system.
Your primary focus is evaluating code structure, design patterns, architectural principles, and overall code organization.

Evaluate the code specifically for:
1. Adherence to SOLID principles
2. Appropriate use of design patterns
3. Separation of concerns
4. Modularity and cohesion
5. Coupling between components
6. Dependency management
7. Error handling and exception flow
8. Testability
9. Extensibility and maintainability
10. Overall architectural consistency

For each issue found, provide:
1. A clear description of the architectural issue
2. The impact on code maintainability and extensibility
3. Specific code location(s) where the issue exists
4. Concrete recommendations for improvement with code examples

Your output should include:
1. A structure score (1-10) for the code
2. Detailed analysis of architectural issues found
3. Prioritized recommendations for improvement
4. Estimated hours for an experienced developer to implement improvements

Format your response with clear sections and markdown formatting for readability.
"""
        super().__init__(model, system_prompt)
    
    async def review(self, code_diff: Dict[str, Any], context: Dict[str, Any], priority: bool = False) -> Dict[str, Any]:
        """Perform an architecture-focused code review."""
        logger.info("Performing architecture review")
        
        # Extract information from code_diff
        file_path = code_diff.get("file_path", "")
        language = code_diff.get("language", "")
        content = code_diff.get("content", "")
        
        # Create prompt for architecture review
        prompt = f"""# Architecture Code Review Request

## File Information
- **File Path**: {file_path}
- **Language**: {language}
- **Priority**: {"High" if priority else "Normal"}

## Code to Review
```{language}
{content}
```

## Context
- **File Type**: {context.get('file_type', 'Unknown')}
- **Complexity**: {context.get('estimated_complexity', 'Unknown')}

## Instructions
Please conduct a thorough architecture review of this code, focusing on:
1. Identifying any architectural issues
2. Assessing design patterns and SOLID principles
3. Providing specific recommendations for improvement
4. Estimating the work hours required to implement improvements

## Response Format
Please structure your response with the following sections:

### SUMMARY
Brief overview of architectural findings

### ARCHITECTURAL ISSUES
Detailed analysis of each architectural issue found

### IMPROVEMENT RECOMMENDATIONS
Prioritized list of architectural improvements

### SCORES:
- Readability: [score] /10
- Efficiency: [score] /10
- Security: [score] /10
- Structure: [score] /10
- Error Handling: [score] /10
- Documentation: [score] /10
- Code Style: [score] /10
- Overall Score: [score] /10

### ESTIMATED HOURS: [hours]
"""
        
        # Generate review
        review_text = await self._generate_review(prompt, "architecture_review")
        
        # Parse scores and estimated hours
        scores = self._parse_scores(review_text)
        estimated_hours = self._parse_estimated_hours(review_text)
        
        # Return review results
        return {
            "scores": scores,
            "feedback": review_text,
            "estimated_hours": estimated_hours
        }


class DocumentationReviewAgent(BaseReviewAgent):
    """
    Specialized agent for documentation code review.
    
    This agent focuses on code documentation, comments,
    docstrings, and overall code documentation quality.
    """
    
    def __init__(self, model):
        """Initialize the documentation review agent."""
        system_prompt = """You are a specialized documentation code review agent within an orchestrated code review system.
Your primary focus is evaluating code documentation, comments, docstrings, and overall documentation quality.

Evaluate the code specifically for:
1. Presence and quality of docstrings
2. Function and method documentation
3. Class and module documentation
4. Inline comments for complex logic
5. API documentation
6. Examples and usage instructions
7. Parameter and return value documentation
8. Exception documentation
9. Consistency in documentation style
10. Overall documentation completeness

For each issue found, provide:
1. A clear description of the documentation issue
2. The impact on code maintainability and usability
3. Specific code location(s) where the issue exists
4. Concrete recommendations for improvement with code examples

Your output should include:
1. A documentation score (1-10) for the code
2. Detailed analysis of documentation issues found
3. Prioritized recommendations for improvement
4. Estimated hours for an experienced developer to implement improvements

Format your response with clear sections and markdown formatting for readability.
"""
        super().__init__(model, system_prompt)
    
    async def review(self, code_diff: Dict[str, Any], context: Dict[str, Any], priority: bool = False) -> Dict[str, Any]:
        """Perform a documentation-focused code review."""
        logger.info("Performing documentation review")
        
        # Extract information from code_diff
        file_path = code_diff.get("file_path", "")
        language = code_diff.get("language", "")
        content = code_diff.get("content", "")
        
        # Create prompt for documentation review
        prompt = f"""# Documentation Code Review Request

## File Information
- **File Path**: {file_path}
- **Language**: {language}
- **Priority**: {"High" if priority else "Normal"}

## Code to Review
```{language}
{content}
```

## Context
- **File Type**: {context.get('file_type', 'Unknown')}
- **Complexity**: {context.get('estimated_complexity', 'Unknown')}

## Instructions
Please conduct a thorough documentation review of this code, focusing on:
1. Identifying any documentation issues
2. Assessing docstrings and comments
3. Providing specific recommendations for improvement
4. Estimating the work hours required to implement improvements

## Response Format
Please structure your response with the following sections:

### SUMMARY
Brief overview of documentation findings

### DOCUMENTATION ISSUES
Detailed analysis of each documentation issue found

### IMPROVEMENT RECOMMENDATIONS
Prioritized list of documentation improvements

### SCORES:
- Readability: [score] /10
- Efficiency: [score] /10
- Security: [score] /10
- Structure: [score] /10
- Error Handling: [score] /10
- Documentation: [score] /10
- Code Style: [score] /10
- Overall Score: [score] /10

### ESTIMATED HOURS: [hours]
"""
        
        # Generate review
        review_text = await self._generate_review(prompt, "documentation_review")
        
        # Parse scores and estimated hours
        scores = self._parse_scores(review_text)
        estimated_hours = self._parse_estimated_hours(review_text)
        
        # Return review results
        return {
            "scores": scores,
            "feedback": review_text,
            "estimated_hours": estimated_hours
        }
