"""
Orchestrator for code review process.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional

from langchain_core.language_models.chat_models import BaseChatModel

from codedog.models import ChangeFile, CodeReview
from codedog.orchestration.coordinator import ReviewCoordinator
from codedog.orchestration.agents import (
    SecurityReviewAgent,
    PerformanceReviewAgent,
    ReadabilityReviewAgent,
    ArchitectureReviewAgent,
    DocumentationReviewAgent
)

logger = logging.getLogger(__name__)

class CodeReviewOrchestrator:
    """
    Orchestrates the code review process using specialized agents.
    
    This class coordinates the work of multiple specialized agents
    to produce a comprehensive code review.
    """
    
    def __init__(self, models: Dict[str, BaseChatModel]):
        """
        Initialize the orchestrator with language models.
        
        Args:
            models: Dictionary of language models for different agents
                   Keys should include: 'coordinator', 'security', 'performance',
                   'readability', 'architecture', 'documentation'
        """
        # Initialize coordinator
        self.coordinator = ReviewCoordinator(model=models.get('coordinator'))
        
        # Initialize specialized agents
        self.agents = {
            'security': SecurityReviewAgent(models.get('security', models.get('default'))),
            'performance': PerformanceReviewAgent(models.get('performance', models.get('default'))),
            'readability': ReadabilityReviewAgent(models.get('readability', models.get('default'))),
            'architecture': ArchitectureReviewAgent(models.get('architecture', models.get('default'))),
            'documentation': DocumentationReviewAgent(models.get('documentation', models.get('default')))
        }
    
    async def review_file(self, change_file: ChangeFile) -> CodeReview:
        """
        Review a single file using the orchestrated process.
        
        Args:
            change_file: The file to review
            
        Returns:
            CodeReview object with the review results
        """
        logger.info(f"Starting orchestrated review for {change_file.full_name}")
        
        # Prepare code diff for review
        code_diff = {
            "file_path": change_file.full_name,
            "language": change_file.language,
            "content": change_file.content
        }
        
        # Step 1: Analyze context
        context = await self.coordinator.analyze_context(code_diff)
        
        # Step 2: Distribute work to specialized agents
        agent_tasks = await self.coordinator.distribute_work(code_diff, context, self.agents)
        
        # Step 3: Wait for all agent tasks to complete
        agent_results = {}
        for agent_name, task in agent_tasks.items():
            try:
                agent_results[agent_name] = await task
            except Exception as e:
                logger.error(f"Error in {agent_name} agent: {e}")
                agent_results[agent_name] = None
        
        # Step 4: Integrate results
        integrated_review = await self.coordinator.integrate_reviews(agent_results)
        
        # Step 5: Generate final report
        code_review = await self.coordinator.generate_report(integrated_review, context)
        
        logger.info(f"Completed orchestrated review for {change_file.full_name}")
        return code_review
    
    async def review_files(self, change_files: List[ChangeFile]) -> List[CodeReview]:
        """
        Review multiple files using the orchestrated process.
        
        Args:
            change_files: List of files to review
            
        Returns:
            List of CodeReview objects with the review results
        """
        logger.info(f"Starting orchestrated review for {len(change_files)} files")
        
        # Create tasks for each file
        tasks = [self.review_file(file) for file in change_files]
        
        # Wait for all tasks to complete
        reviews = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        code_reviews = []
        for i, review in enumerate(reviews):
            if isinstance(review, Exception):
                logger.error(f"Error reviewing {change_files[i].full_name}: {review}")
                # Create an error review
                code_reviews.append(CodeReview(
                    file=change_files[i],
                    review=f"Error during review: {str(review)}"
                ))
            else:
                code_reviews.append(review)
        
        logger.info(f"Completed orchestrated review for {len(change_files)} files")
        return code_reviews
