"""
Coordinator module for orchestrating the code review process.
"""

import logging
from typing import Dict, List, Any, Optional
import asyncio

from codedog.models import ChangeFile, CodeReview
from codedog.utils.code_evaluator import log_llm_interaction

logger = logging.getLogger(__name__)

class ReviewCoordinator:
    """
    Coordinates the code review process between specialized agents.
    
    This class is responsible for:
    1. Analyzing the context of code changes
    2. Distributing work to specialized agents
    3. Integrating results from different agents
    4. Generating the final report
    """
    
    def __init__(self, model=None):
        """
        Initialize the coordinator.
        
        Args:
            model: Optional language model for context analysis and integration
        """
        self.model = model
        
    async def analyze_context(self, code_diff: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the context of code changes to guide specialized reviews.
        
        Args:
            code_diff: The code diff to analyze
            
        Returns:
            Dict containing context information for specialized agents
        """
        logger.info("Analyzing context for code review")
        
        # Extract basic information
        file_path = code_diff.get("file_path", "")
        language = code_diff.get("language", "")
        content = code_diff.get("content", "")
        
        # Determine file type and primary concerns
        file_type = self._determine_file_type(file_path, language)
        primary_concerns = self._identify_primary_concerns(file_type, content)
        
        # Create context object
        context = {
            "file_path": file_path,
            "language": language,
            "file_type": file_type,
            "primary_concerns": primary_concerns,
            "estimated_complexity": self._estimate_complexity(content),
        }
        
        logger.info(f"Context analysis complete: {context}")
        return context
    
    async def distribute_work(self, code_diff: Dict[str, Any], context: Dict[str, Any], agents: Dict[str, Any]) -> Dict[str, asyncio.Task]:
        """
        Distribute work to specialized agents based on context.
        
        Args:
            code_diff: The code diff to review
            context: The context information
            agents: Dictionary of specialized agents
            
        Returns:
            Dictionary of agent tasks
        """
        logger.info("Distributing work to specialized agents")
        
        tasks = {}
        for agent_name, agent in agents.items():
            # Determine if this agent should be prioritized based on context
            priority = agent_name in context.get("primary_concerns", [])
            
            # Create task for this agent
            tasks[agent_name] = asyncio.create_task(
                agent.review(code_diff, context, priority=priority)
            )
            
        return tasks
    
    async def integrate_reviews(self, reviews: Dict[str, Any]) -> Dict[str, Any]:
        """
        Integrate reviews from specialized agents into a cohesive result.
        
        Args:
            reviews: Dictionary of reviews from specialized agents
            
        Returns:
            Integrated review
        """
        logger.info("Integrating specialized reviews")
        
        # Combine scores from different agents with appropriate weighting
        scores = self._combine_scores(reviews)
        
        # Combine feedback from different agents
        feedback = self._combine_feedback(reviews)
        
        # Create integrated review
        integrated_review = {
            "scores": scores,
            "feedback": feedback,
            "estimated_hours": self._calculate_estimated_hours(reviews),
        }
        
        return integrated_review
    
    async def generate_report(self, integrated_review: Dict[str, Any], context: Dict[str, Any]) -> CodeReview:
        """
        Generate the final code review report.
        
        Args:
            integrated_review: The integrated review from specialized agents
            context: The context information
            
        Returns:
            CodeReview object
        """
        logger.info("Generating final code review report")
        
        # Format the report
        report = self._format_report(integrated_review, context)
        
        # Create CodeReview object
        code_review = CodeReview(
            file=ChangeFile(
                full_name=context.get("file_path", ""),
                language=context.get("language", ""),
            ),
            review=report
        )
        
        return code_review
    
    def _determine_file_type(self, file_path: str, language: str) -> str:
        """Determine the type of file based on path and language."""
        if not file_path:
            return "unknown"
            
        if language.lower() in ["python", "py"]:
            if "test" in file_path.lower():
                return "python_test"
            elif "model" in file_path.lower():
                return "python_model"
            elif "view" in file_path.lower():
                return "python_view"
            elif "controller" in file_path.lower() or "handler" in file_path.lower():
                return "python_controller"
            else:
                return "python_general"
                
        # Add similar logic for other languages
        return language.lower() if language else "unknown"
    
    def _identify_primary_concerns(self, file_type: str, content: str) -> List[str]:
        """Identify primary concerns based on file type and content."""
        concerns = []
        
        # File type based concerns
        if file_type.startswith("python_test"):
            concerns.extend(["readability", "architecture"])
        elif file_type.startswith("python_model"):
            concerns.extend(["performance", "security"])
        elif file_type.startswith("python_controller"):
            concerns.extend(["security", "architecture"])
            
        # Content based concerns
        if "password" in content.lower() or "auth" in content.lower():
            concerns.append("security")
        if "database" in content.lower() or "query" in content.lower():
            concerns.append("performance")
        if "TODO" in content or "FIXME" in content:
            concerns.append("documentation")
            
        # Ensure uniqueness
        return list(set(concerns))
    
    def _estimate_complexity(self, content: str) -> str:
        """Estimate the complexity of the code."""
        # Simple heuristic based on length and structure
        lines = content.split("\n")
        line_count = len(lines)
        
        if line_count < 50:
            return "low"
        elif line_count < 200:
            return "medium"
        else:
            return "high"
    
    def _combine_scores(self, reviews: Dict[str, Any]) -> Dict[str, float]:
        """Combine scores from different specialized agents."""
        combined_scores = {
            "readability": 0,
            "efficiency": 0,
            "security": 0,
            "structure": 0,
            "error_handling": 0,
            "documentation": 0,
            "code_style": 0,
            "overall_score": 0
        }
        
        # Define weights for different agent types
        weights = {
            "security": {"security": 2.0, "overall_score": 1.2},
            "performance": {"efficiency": 2.0, "overall_score": 1.2},
            "readability": {"readability": 2.0, "code_style": 1.5, "overall_score": 1.2},
            "architecture": {"structure": 2.0, "error_handling": 1.5, "overall_score": 1.2},
            "documentation": {"documentation": 2.0, "overall_score": 1.0}
        }
        
        # Track weight totals for normalization
        weight_totals = {key: 0 for key in combined_scores}
        
        # Combine scores with weighting
        for agent_type, review in reviews.items():
            if not review or "scores" not in review:
                continue
                
            for score_type, score in review["scores"].items():
                if score_type in combined_scores:
                    # Get weight for this agent and score type
                    weight = weights.get(agent_type, {}).get(score_type, 1.0)
                    
                    # Add weighted score
                    combined_scores[score_type] += score * weight
                    weight_totals[score_type] += weight
        
        # Normalize scores
        for score_type in combined_scores:
            if weight_totals[score_type] > 0:
                combined_scores[score_type] /= weight_totals[score_type]
                # Round to one decimal place
                combined_scores[score_type] = round(combined_scores[score_type], 1)
            else:
                combined_scores[score_type] = 5.0  # Default score
        
        return combined_scores
    
    def _combine_feedback(self, reviews: Dict[str, Any]) -> str:
        """Combine feedback from different specialized agents."""
        combined_feedback = []
        
        # Add feedback from each agent with appropriate headers
        for agent_type, review in reviews.items():
            if not review or "feedback" not in review:
                continue
                
            # Add header for this agent's feedback
            header = f"## {agent_type.capitalize()} Review"
            combined_feedback.append(header)
            
            # Add the feedback
            combined_feedback.append(review["feedback"])
            
            # Add separator
            combined_feedback.append("")
        
        return "\n".join(combined_feedback)
    
    def _calculate_estimated_hours(self, reviews: Dict[str, Any]) -> float:
        """Calculate estimated hours from specialized agent reviews."""
        # Get estimates from each agent
        estimates = []
        for review in reviews.values():
            if review and "estimated_hours" in review:
                estimates.append(review["estimated_hours"])
        
        # Calculate final estimate
        if not estimates:
            return 0.0
            
        # Use the maximum estimate as the base
        base_estimate = max(estimates)
        
        # Add 10% for each additional estimate to account for integration work
        additional = 0.1 * base_estimate * (len(estimates) - 1)
        
        return round(base_estimate + additional, 1)
    
    def _format_report(self, integrated_review: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Format the final code review report."""
        scores = integrated_review.get("scores", {})
        feedback = integrated_review.get("feedback", "")
        estimated_hours = integrated_review.get("estimated_hours", 0.0)
        
        # Create report header
        report = [
            f"# Code Review for {context.get('file_path', 'Unknown File')}",
            "",
            f"**Language:** {context.get('language', 'Unknown')}",
            f"**Complexity:** {context.get('estimated_complexity', 'Unknown')}",
            f"**Estimated Hours:** {estimated_hours}",
            "",
            "## Scores",
            "",
            "| Category | Score |",
            "|----------|-------|",
        ]
        
        # Add scores
        for category, score in scores.items():
            # Convert category name to title case with spaces
            category_name = category.replace("_", " ").title()
            report.append(f"| {category_name} | {score:.1f} / 10 |")
        
        # Add feedback
        report.append("")
        report.append("## Detailed Feedback")
        report.append("")
        report.append(feedback)
        
        return "\n".join(report)
