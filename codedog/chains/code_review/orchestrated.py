"""
Orchestrated code review chain using specialized agents.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from langchain_core.language_models import BaseLanguageModel
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.chains.base import Chain
from pydantic import Field

from codedog.models import ChangeFile, CodeReview, PullRequest
from codedog.processors import PullRequestProcessor
from codedog.orchestration.orchestrator import CodeReviewOrchestrator


class OrchestratedCodeReviewChain(Chain):
    """
    Orchestrated code review chain using specialized agents.
    
    This chain uses the orchestration architecture to coordinate
    multiple specialized agents for a comprehensive code review.
    """
    
    orchestrator: CodeReviewOrchestrator = Field(exclude=True)
    """Orchestrator for code review."""
    
    processor: PullRequestProcessor = Field(
        exclude=True, default_factory=PullRequestProcessor.build
    )
    """PR data processor."""
    
    _input_keys: List[str] = ["pull_request"]
    _output_keys: List[str] = ["code_reviews"]

    @property
    def _chain_type(self) -> str:
        return "orchestrated_code_review_chain"

    @property
    def input_keys(self) -> List[str]:
        """Will be whatever keys the prompt expects.

        :meta private:
        """
        return self._input_keys

    @property
    def output_keys(self) -> List[str]:
        """Will always return text key.

        :meta private:
        """
        return self._output_keys

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """
        Synchronous call is not supported for orchestrated review.
        Use async call instead.
        """
        raise NotImplementedError(
            "Orchestrated code review only supports async calls. Use acall instead."
        )

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """
        Perform orchestrated code review asynchronously.
        """
        _run_manager = run_manager or AsyncCallbackManagerForChainRun.get_noop_manager()
        await _run_manager.on_text(inputs["pull_request"].json() + "\n")

        pr: PullRequest = inputs["pull_request"]
        code_files: List[ChangeFile] = self.processor.get_diff_code_files(pr)

        # Use orchestrator to review files
        code_reviews = await self.orchestrator.review_files(code_files)

        return {"code_reviews": code_reviews}

    @classmethod
    def from_llms(
        cls,
        *,
        models: Dict[str, BaseLanguageModel],
        **kwargs,
    ) -> OrchestratedCodeReviewChain:
        """
        Create an orchestrated code review chain from language models.
        
        Args:
            models: Dictionary of language models for different agents
                   Keys should include: 'coordinator', 'security', 'performance',
                   'readability', 'architecture', 'documentation', 'default'
                   
        Returns:
            OrchestratedCodeReviewChain instance
        """
        # Ensure we have a default model
        if "default" not in models and len(models) > 0:
            # Use the first model as default
            models["default"] = next(iter(models.values()))
            
        # Create orchestrator
        orchestrator = CodeReviewOrchestrator(models)
        
        return cls(
            orchestrator=orchestrator,
            processor=PullRequestProcessor(),
        )
