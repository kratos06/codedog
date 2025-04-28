"""
Orchestration module for CodeDog.

This module contains the orchestration architecture for coordinating
specialized agents in the code review process.
"""

from codedog.orchestration.coordinator import ReviewCoordinator
from codedog.orchestration.orchestrator import CodeReviewOrchestrator
from codedog.orchestration.agents import (
    SecurityReviewAgent,
    PerformanceReviewAgent,
    ReadabilityReviewAgent,
    ArchitectureReviewAgent,
    DocumentationReviewAgent
)

__all__ = [
    "ReviewCoordinator",
    "CodeReviewOrchestrator",
    "SecurityReviewAgent",
    "PerformanceReviewAgent",
    "ReadabilityReviewAgent",
    "ArchitectureReviewAgent",
    "DocumentationReviewAgent"
]
