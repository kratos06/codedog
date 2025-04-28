"""
Base agent class for specialized code review agents.
"""

import logging
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from codedog.utils.code_evaluator import log_llm_interaction

logger = logging.getLogger(__name__)

class BaseReviewAgent(ABC):
    """
    Base class for specialized code review agents.
    
    This abstract class defines the interface for all specialized agents
    and provides common functionality.
    """
    
    def __init__(self, model: BaseChatModel, system_prompt: str):
        """
        Initialize the agent.
        
        Args:
            model: Language model to use for reviews
            system_prompt: System prompt for this specialized agent
        """
        self.model = model
        self.system_prompt = system_prompt
        
    @abstractmethod
    async def review(self, code_diff: Dict[str, Any], context: Dict[str, Any], priority: bool = False) -> Dict[str, Any]:
        """
        Review the code diff with this specialized agent.
        
        Args:
            code_diff: The code diff to review
            context: Context information from the coordinator
            priority: Whether this agent should be prioritized
            
        Returns:
            Dictionary containing review results
        """
        pass
    
    async def _generate_review(self, prompt: str, interaction_type: str = "specialized_review") -> str:
        """
        Generate a review using the language model.
        
        Args:
            prompt: The prompt to send to the model
            interaction_type: Type of interaction for logging
            
        Returns:
            The generated review text
        """
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt)
        ]
        
        # Log the prompt
        log_llm_interaction(prompt, "", interaction_type=interaction_type)
        
        # Call the model
        response = await self.model.agenerate(messages=[messages])
        generated_text = response.generations[0][0].text
        
        # Log the response
        log_llm_interaction("", generated_text, interaction_type=interaction_type)
        
        return generated_text
    
    def _parse_scores(self, review_text: str) -> Dict[str, float]:
        """
        Parse scores from the review text.
        
        Args:
            review_text: The review text to parse
            
        Returns:
            Dictionary of scores
        """
        import re
        
        # Default scores
        scores = {
            "readability": 5.0,
            "efficiency": 5.0,
            "security": 5.0,
            "structure": 5.0,
            "error_handling": 5.0,
            "documentation": 5.0,
            "code_style": 5.0,
            "overall_score": 5.0
        }
        
        # Try to extract scores using regex
        try:
            # Look for score section
            score_section_match = re.search(r'#{1,3}\s*(?:SCORES|RATINGS):\s*([\s\S]*?)(?=#{1,3}|$)', review_text, re.IGNORECASE)
            if score_section_match:
                score_section = score_section_match.group(1)
                
                # Extract individual scores
                for category in scores.keys():
                    # Convert category to regex pattern (e.g., "error_handling" -> "Error\s*Handling")
                    category_pattern = category.replace('_', r'\s*').title()
                    score_match = re.search(rf'[-*]\s*{category_pattern}:\s*(\d+(?:\.\d+)?)\s*/\s*10', score_section, re.IGNORECASE)
                    if score_match:
                        scores[category] = float(score_match.group(1))
                
                # Try to extract overall score if not found above
                overall_match = re.search(r'[-*]\s*(?:Final\s+)?Overall(?:\s+Score)?:\s*(\d+(?:\.\d+)?)\s*/\s*10', score_section, re.IGNORECASE)
                if overall_match:
                    scores["overall_score"] = float(overall_match.group(1))
        except Exception as e:
            logger.warning(f"Error parsing scores: {e}")
        
        return scores
    
    def _parse_estimated_hours(self, review_text: str) -> float:
        """
        Parse estimated hours from the review text.
        
        Args:
            review_text: The review text to parse
            
        Returns:
            Estimated hours as a float
        """
        import re
        
        # Default estimate
        estimated_hours = 0.0
        
        # Try to extract estimated hours using regex
        try:
            # Look for estimated hours
            hours_match = re.search(r'(?:estimated|approximate)\s+(?:work\s*)?hours?:?\s*(\d+(?:\.\d+)?)', review_text, re.IGNORECASE)
            if hours_match:
                estimated_hours = float(hours_match.group(1))
        except Exception as e:
            logger.warning(f"Error parsing estimated hours: {e}")
        
        return estimated_hours
