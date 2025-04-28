"""
Factory for creating code review chains.
"""

from typing import Dict, Optional

from langchain_core.language_models import BaseLanguageModel

from codedog.chains.code_review.base import CodeReviewChain
from codedog.chains.code_review.orchestrated import OrchestratedCodeReviewChain


class CodeReviewChainFactory:
    """
    Factory for creating code review chains.
    
    This factory supports creating both traditional and orchestrated
    code review chains based on configuration.
    """
    
    @staticmethod
    def create_chain(
        llm: BaseLanguageModel,
        use_orchestration: bool = False,
        models: Optional[Dict[str, BaseLanguageModel]] = None,
        **kwargs
    ):
        """
        Create a code review chain.
        
        Args:
            llm: Language model for traditional chain or default model for orchestrated chain
            use_orchestration: Whether to use orchestration
            models: Dictionary of language models for different agents (for orchestration)
            **kwargs: Additional arguments for chain creation
            
        Returns:
            CodeReviewChain or OrchestratedCodeReviewChain instance
        """
        if use_orchestration:
            # Prepare models dictionary
            if models is None:
                models = {"default": llm}
            elif "default" not in models:
                models["default"] = llm
                
            # Create orchestrated chain
            return OrchestratedCodeReviewChain.from_llms(
                models=models,
                **kwargs
            )
        else:
            # Create traditional chain
            return CodeReviewChain.from_llm(
                llm=llm,
                **kwargs
            )
