import asyncio
import json
import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import re
import logging  # Add logging import
import os
import random
import time
import tenacity
from tenacity import retry, stop_after_attempt, wait_exponential
import math
import tiktoken  # 用于精确计算token数量

# Function to log LLM inputs and outputs to a single file
def log_llm_interaction(prompt, response, interaction_type="default", extra_info=None):
    """
    Log LLM prompts and responses to a single LLM_interactions.log file

    Args:
        prompt: The prompt sent to the LLM
        response: The response received from the LLM
        interaction_type: A label to identify the type of interaction (e.g., "file_evaluation", "summary")
        extra_info: Optional additional information to log (e.g., commit details for whole-commit evaluation)
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Log both prompt and response to the same file
    with open("logs/LLM_interactions.log", "a", encoding="utf-8") as f:
        f.write(f"\n\n{'='*80}\n")
        f.write(f"TIMESTAMP: {timestamp}\n")
        f.write(f"TYPE: {interaction_type}\n")

        # If we have extra info, log it
        if extra_info:
            f.write(f"INFO: {extra_info}\n")

        f.write(f"{'='*80}\n\n")

        # If we have a prompt, log it
        if prompt:
            f.write(f"[PROMPT]\n{'-'*50}\n")
            f.write(prompt)
            f.write("\n\n")

        # If we have a response, log it
        if response:
            f.write(f"[RESPONSE]\n{'-'*50}\n")
            f.write(response)
            f.write("\n\n")

        # Add a separator at the end
        f.write(f"{'='*80}\n")

# 导入 grimoire 模板
from codedog.templates.grimoire_en import CODE_SUGGESTION
from codedog.templates.grimoire_cn import GrimoireCn
# 导入优化的代码评审prompt
from codedog.templates.optimized_code_review_prompt import (
    SYSTEM_PROMPT,
    CODE_REVIEW_PROMPT,
    LANGUAGE_SPECIFIC_CONSIDERATIONS
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from codedog.utils.git_log_analyzer import CommitInfo


class CodeEvaluation(BaseModel):
    """Structured output for code evaluation"""
    readability: int = Field(description="Code readability score (1-10)", ge=1, le=10)
    efficiency: int = Field(description="Code efficiency and performance score (1-10)", ge=1, le=10)
    security: int = Field(description="Code security score (1-10)", ge=1, le=10)
    structure: int = Field(description="Code structure and design score (1-10)", ge=1, le=10)
    error_handling: int = Field(description="Error handling score (1-10)", ge=1, le=10)
    documentation: int = Field(description="Documentation and comments score (1-10)", ge=1, le=10)
    code_style: int = Field(description="Code style score (1-10)", ge=1, le=10)
    overall_score: float = Field(description="Overall score (1-10)", ge=1, le=10)
    effective_code_lines: int = Field(description="Number of lines with actual logic/functionality changes", default=0)
    non_effective_code_lines: int = Field(description="Number of lines with formatting, whitespace, comment changes", default=0)
    estimated_hours: float = Field(description="Estimated working hours for an experienced programmer (5-10+ years)", default=0.0)
    comments: str = Field(description="Evaluation comments and improvement suggestions")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CodeEvaluation":
        """Create a CodeEvaluation instance from a dictionary, handling float scores."""
        # Convert float scores to integers for all score fields except overall_score
        score_fields = ["readability", "efficiency", "security", "structure",
                       "error_handling", "documentation", "code_style"]

        # Log the original data
        logger.info(f"Creating CodeEvaluation from data: {data}")
        print(f"DEBUG: Creating CodeEvaluation from data: {data}")

        # Make a copy of the data to avoid modifying the original
        data_copy = data.copy()

        for field in score_fields:
            if field in data_copy and isinstance(data_copy[field], float):
                # Log the conversion
                logger.info(f"Converting {field} from float {data_copy[field]} to int {round(data_copy[field])}")
                data_copy[field] = round(data_copy[field])

        # Create the instance
        instance = cls(**data_copy)

        # Log the created instance
        logger.info(f"Created CodeEvaluation instance: {instance}")
        print(f"DEBUG: Created CodeEvaluation instance: {instance}")

        return instance


@dataclass(frozen=True)  # Make it immutable and hashable
class FileEvaluationResult:
    """文件评价结果"""
    file_path: str
    commit_hash: str
    commit_message: str
    date: datetime
    author: str
    evaluation: CodeEvaluation


class TokenBucket:
    """Token bucket for rate limiting with improved algorithm and better concurrency handling"""
    def __init__(self, tokens_per_minute: int = 10000, update_interval: float = 1.0):
        self.tokens_per_minute = tokens_per_minute
        self.update_interval = update_interval
        self.tokens = tokens_per_minute
        self.last_update = time.time()
        self.lock = asyncio.Lock()
        self.total_tokens_used = 0  # 统计总共使用的令牌数
        self.total_wait_time = 0.0  # 统计总共等待的时间
        self.pending_requests = []  # 待处理的请求队列
        self.request_count = 0  # 请求计数器

    async def get_tokens(self, requested_tokens: int) -> float:
        """Get tokens from the bucket. Returns the wait time needed."""
        # 生成唯一的请求ID
        request_id = self.request_count
        self.request_count += 1

        # 创建一个事件，用于通知请求何时可以继续
        event = asyncio.Event()
        wait_time = 0.0

        async with self.lock:
            now = time.time()
            time_passed = now - self.last_update

            # Replenish tokens
            self.tokens = min(
                self.tokens_per_minute,
                self.tokens + (time_passed * self.tokens_per_minute / 60.0)
            )
            self.last_update = now

            # 检查是否有足够的令牌
            if self.tokens >= requested_tokens:
                # 有足够的令牌，直接处理
                self.tokens -= requested_tokens
                self.total_tokens_used += requested_tokens
                return 0.0
            else:
                # 没有足够的令牌，需要等待
                # 先消耗掉当前所有可用的令牌
                available_tokens = self.tokens
                self.tokens = 0
                self.total_tokens_used += available_tokens

                # 计算还需要多少令牌
                tokens_still_needed = requested_tokens - available_tokens

                # 计算需要等待的时间
                wait_time = (tokens_still_needed * 60.0 / self.tokens_per_minute)

                # 添加一些随机性，避免雇佯效应
                wait_time *= (1 + random.uniform(0, 0.1))

                # 更新统计信息
                self.total_wait_time += wait_time

                # 将请求添加到队列中，包含请求ID、所需令牌数、事件和计算出的等待时间
                self.pending_requests.append((request_id, tokens_still_needed, event, wait_time))

                # 按等待时间排序，使小请求先处理
                self.pending_requests.sort(key=lambda x: x[3])

                # 启动令牌补充任务
                asyncio.create_task(self._replenish_tokens())

        # 等待事件触发
        await event.wait()
        return wait_time

    async def _replenish_tokens(self):
        """Continuously replenish tokens and process pending requests"""
        while True:
            # 等待一小段时间
            await asyncio.sleep(0.1)

            async with self.lock:
                # 如果没有待处理的请求，则退出
                if not self.pending_requests:
                    break

                # 计算经过的时间和新生成的令牌
                now = time.time()
                time_passed = now - self.last_update
                new_tokens = time_passed * self.tokens_per_minute / 60.0

                # 更新令牌数量和时间
                self.tokens += new_tokens
                self.last_update = now

                # 处理待处理的请求
                i = 0
                while i < len(self.pending_requests):
                    _, tokens_needed, event, _ = self.pending_requests[i]

                    # 如果有足够的令牌，则处理这个请求
                    if self.tokens >= tokens_needed:
                        self.tokens -= tokens_needed
                        # 触发事件，通知请求可以继续
                        event.set()
                        # 从待处理列表中移除这个请求
                        self.pending_requests.pop(i)
                    else:
                        # 没有足够的令牌，移动到下一个请求
                        i += 1

                # 如果所有请求都处理完毕，则退出
                if not self.pending_requests:
                    break

    def get_stats(self) -> Dict[str, float]:
        """获取令牌桶的使用统计信息"""
        now = time.time()
        time_passed = now - self.last_update

        # 计算当前可用令牌，考虑从上次更新到现在的时间内生成的令牌
        current_tokens = min(
            self.tokens_per_minute,
            self.tokens + (time_passed * self.tokens_per_minute / 60.0)
        )

        # 计算当前使用率
        usage_rate = 0
        if self.total_tokens_used > 0:
            elapsed_time = now - self.last_update + self.total_wait_time
            if elapsed_time > 0:
                usage_rate = self.total_tokens_used / (elapsed_time / 60.0)

        # 计算当前并发请求数
        pending_requests = len(self.pending_requests)

        # 计算估计的恢复时间
        recovery_time = 0
        if pending_requests > 0 and self.tokens_per_minute > 0:
            # 获取所有待处理请求的总令牌数
            total_pending_tokens = sum(tokens for _, tokens, _, _ in self.pending_requests)
            # 计算恢复时间
            recovery_time = max(0, (total_pending_tokens - current_tokens) * 60.0 / self.tokens_per_minute)

        return {
            "tokens_per_minute": self.tokens_per_minute,
            "current_tokens": current_tokens,
            "total_tokens_used": self.total_tokens_used,
            "total_wait_time": self.total_wait_time,
            "average_wait_time": self.total_wait_time / max(1, self.total_tokens_used / 1000),  # 每1000个令牌的平均等待时间
            "pending_requests": pending_requests,
            "usage_rate": usage_rate,  # 实际使用率（令牌/分钟）
            "recovery_time": recovery_time  # 估计的恢复时间（秒）
        }


def count_tokens(text: str, model_name: str = "gpt-3.5-turbo") -> int:
    """精确计算文本的token数量

    Args:
        text: 要计算的文本
        model_name: 模型名称，默认为 gpt-3.5-turbo

    Returns:
        int: token数量
    """
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        # 如果模型不在tiktoken的列表中，使用默认编码
        encoding = tiktoken.get_encoding("cl100k_base")

    # 计算token数量
    tokens = encoding.encode(text)
    return len(tokens)


def save_diff_content(file_path: str, diff_content: str, estimated_tokens: int, actual_tokens: int = None):
    """将diff内容保存到中间文件中

    Args:
        file_path: 文件路径
        diff_content: diff内容
        estimated_tokens: 估算的token数量
        actual_tokens: 实际的token数量，如果为None则会计算
    """
    # 创建diffs目录，如果不存在
    os.makedirs("diffs", exist_ok=True)

    # 生成安全的文件名
    safe_name = re.sub(r'[^\w\-_.]', '_', file_path)
    output_path = f"diffs/{safe_name}.diff"

    # 计算实际token数量，如果没有提供
    if actual_tokens is None:
        actual_tokens = count_tokens(diff_content)

    # 添加元数据到diff内容中
    metadata = f"""# File: {file_path}
# Estimated tokens: {estimated_tokens}
# Actual tokens: {actual_tokens}
# Token ratio (actual/estimated): {actual_tokens/estimated_tokens:.2f}
# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

"""

    # 写入文件
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(metadata + diff_content)

    logger.info(f"Saved diff content to {output_path} (estimated: {estimated_tokens}, actual: {actual_tokens} tokens)")

    # 如果实际token数量远远超过估计值，记录警告
    if actual_tokens > estimated_tokens * 1.5:
        logger.warning(f"Warning: Actual token count ({actual_tokens}) significantly exceeds estimated value ({estimated_tokens})")


class DiffEvaluator:
    """代码差异评价器"""

    def __init__(self, model: BaseChatModel, tokens_per_minute: int = 9000, max_concurrent_requests: int = 3,
                 save_diffs: bool = False):
        """
        初始化评价器

        Args:
            model: 用于评价代码的语言模型
            tokens_per_minute: 每分钟令牌数量限制，默认为9000
            max_concurrent_requests: 最大并发请求数，默认为3
            save_diffs: 是否保存diff内容到中间文件，默认为False
        """
        self.model = model
        self.parser = PydanticOutputParser(pydantic_object=CodeEvaluation)
        self.save_diffs = save_diffs  # 新增参数，控制是否保存diff内容

        # 获取模型名称，用于计算token
        self.model_name = getattr(model, "model_name", "gpt-3.5-turbo")

        # Rate limiting settings - 自适应速率控制
        self.initial_tokens_per_minute = tokens_per_minute  # 初始令牌生成速率
        self.token_bucket = TokenBucket(tokens_per_minute=self.initial_tokens_per_minute)  # 留出缓冲
        self.MIN_REQUEST_INTERVAL = 1.0  # 请求之间的最小间隔
        self.MAX_CONCURRENT_REQUESTS = max_concurrent_requests  # 最大并发请求数
        self.request_semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_REQUESTS)
        self._last_request_time = 0

        # 自适应控制参数
        self.rate_limit_backoff_factor = 1.5  # 遇到速率限制时的退避因子
        self.rate_limit_recovery_factor = 1.2  # 成功一段时间后的恢复因子
        self.consecutive_failures = 0  # 连续失败次数
        self.consecutive_successes = 0  # 连续成功次数
        self.success_threshold = 10  # 连续成功多少次后尝试恢复速率
        self.rate_limit_errors = 0  # 速率限制错误计数
        self.last_rate_adjustment_time = time.time()  # 上次调整速率的时间

        # 缓存设置
        self.cache = {}  # 简单的内存缓存 {file_hash: evaluation_result}
        self.cache_hits = 0  # 缓存命中次数

        # 创建diffs目录，如果需要保存diff内容
        if self.save_diffs:
            os.makedirs("diffs", exist_ok=True)

        # System prompt - 使用优化的系统提示
        self.system_prompt = """# ROLE AND OBJECTIVE
You are a senior code reviewer with 15+ years of experience across multiple programming languages and frameworks. Your task is to provide a thorough, objective evaluation of code quality and estimate the effort required to implement the changes.

# EVALUATION DIMENSIONS
Evaluate the code on these dimensions, scoring each from 1-10 (10 being highest):

1. Readability (1-10): Code clarity, naming conventions, consistent formatting
2. Efficiency (1-10): Algorithmic efficiency, resource usage, performance considerations
3. Security (1-10): Protection against vulnerabilities, input validation, secure coding practices
4. Structure (1-10): Architecture, modularity, separation of concerns, SOLID principles
5. Error Handling (1-10): Robust error handling, edge cases, graceful failure
6. Documentation (1-10): Comments, docstrings, self-documenting code
7. Code Style (1-10): Adherence to language/framework conventions and best practices
8. Overall Score (1-10): Comprehensive evaluation considering all dimensions

# CODE CHANGE CLASSIFICATION
When evaluating code changes (especially in diff format), carefully distinguish between:

## Non-Effective Changes (Should NOT count significantly toward working hours)
- Whitespace adjustments (spaces, tabs, line breaks)
- Indentation fixes without logic changes
- Comment additions or modifications without code changes
- Import reordering or reorganization
- Variable/function/class renaming without behavior changes
- Moving code without changing logic
- Trivial refactoring that doesn't improve performance or maintainability
- Code reformatting (line wrapping, bracket placement)
- Changing string quotes (single to double quotes)
- Adding/removing trailing commas
- Changing code style to match linter rules
- Removing unused imports or variables

## Effective Changes (SHOULD count toward working hours)
- Logic modifications that alter program behavior
- Functionality additions or removals
- Algorithm changes or optimizations
- Bug fixes that correct actual issues
- API changes (parameters, return types, etc.)
- Data structure modifications
- Performance optimizations
- Security vulnerability fixes
- Error handling improvements
- Complex refactoring that maintains behavior but improves code quality

# WORKING HOURS ESTIMATION GUIDELINES
When estimating the time an experienced programmer (5-10+ years) would need:

1. For purely non-effective changes:
   - 0.1-0.2 hours for small files
   - 0.3-0.5 hours for large files with extensive formatting

2. For effective changes, consider:
   - Complexity of the logic (simple, moderate, complex)
   - Domain knowledge required (general, specialized, expert)
   - Testing requirements (minimal, moderate, extensive)
   - Integration complexity (isolated, moderate dependencies, highly coupled)

3. Time components to include in your estimate:
   - Understanding the existing code
   - Designing the solution
   - Implementing the changes
   - Testing and debugging
   - Documentation and code review

4. Provide a realistic estimate that reflects the actual work required, not just the line count.

# LANGUAGE-SPECIFIC CONSIDERATIONS
- For Python: Consider PEP 8 compliance, type hints, docstrings
- For JavaScript/TypeScript: Consider ES6+ features, typing, framework conventions
- For Java: Consider OOP principles, exception handling, Java conventions
- For C/C++: Consider memory management, performance, platform considerations
- For other languages: Apply relevant best practices and conventions

3. When reviewing diff format code:
   - Pay attention to both added (+) and removed (-) lines to understand the complete change
   - Evaluate the net effect of the changes, not just individual lines
   - Consider the context of the entire file when evaluating specific changes

4. If you cannot evaluate the code:
   - Assign a default score of 5 for each dimension
   - Explain why evaluation wasn't possible
   - Estimate minimal working hours (0.25) for changes that cannot be properly evaluated

Always return valid JSON format with all required fields."""

        # 添加JSON输出指令
        self.json_output_instruction = """
# OUTPUT FORMAT
Return your evaluation in valid JSON format with the following structure:

```json
{
  "readability": score,
  "efficiency": score,
  "security": score,
  "structure": score,
  "error_handling": score,
  "documentation": score,
  "code_style": score,
  "overall_score": score,
  "effective_code_lines": number,
  "non_effective_code_lines": number,
  "estimated_hours": number,
  "comments": "detailed analysis with specific observations and recommendations"
}
```

## JSON Output Guidelines:
1. All scores MUST be integers or decimals between 1-10
2. The overall_score should reflect the weighted importance of all dimensions
3. effective_code_lines should count ONLY changes that affect behavior or functionality
4. non_effective_code_lines should count formatting, style, and cosmetic changes
5. estimated_hours should be a realistic estimate for an experienced programmer (5-10+ years)
6. comments should include:
   - Specific observations about code quality
   - Concrete recommendations for improvement
   - Explanation of effective vs. non-effective changes
   - Justification for your working hours estimate
   - Any security concerns or performance issues
   - Suggestions for better practices or patterns

IMPORTANT: Ensure your response is valid JSON that can be parsed programmatically. Do not include explanatory text outside the JSON structure.
"""

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=10),
        retry=tenacity.retry_if_exception_type(Exception)
    )
    def _calculate_file_hash(self, diff_content: str) -> str:
        """计算文件差异内容的哈希值，用于缓存"""
        return hashlib.md5(diff_content.encode('utf-8')).hexdigest()

    def _adjust_rate_limits(self, is_rate_limited: bool = False):
        """根据API响应动态调整速率限制

        Args:
            is_rate_limited: 是否遇到了速率限制错误
        """
        now = time.time()

        # 如果遇到速率限制错误
        if is_rate_limited:
            self.consecutive_failures += 1
            self.consecutive_successes = 0
            self.rate_limit_errors += 1

            # 减少令牌生成速率
            new_rate = self.token_bucket.tokens_per_minute / self.rate_limit_backoff_factor
            logger.warning(f"Rate limit encountered, reducing token generation rate: {self.token_bucket.tokens_per_minute:.0f} -> {new_rate:.0f} tokens/min")
            print(f"⚠️ Rate limit encountered, reducing request rate: {self.token_bucket.tokens_per_minute:.0f} -> {new_rate:.0f} tokens/min")
            self.token_bucket.tokens_per_minute = new_rate

            # 增加最小请求间隔
            self.MIN_REQUEST_INTERVAL *= self.rate_limit_backoff_factor
            logger.warning(f"Increasing minimum request interval: {self.MIN_REQUEST_INTERVAL:.2f}s")

            # 减少最大并发请求数，但不少于1
            if self.MAX_CONCURRENT_REQUESTS > 1:
                self.MAX_CONCURRENT_REQUESTS = max(1, self.MAX_CONCURRENT_REQUESTS - 1)
                self.request_semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_REQUESTS)
                logger.warning(f"Reducing maximum concurrent requests: {self.MAX_CONCURRENT_REQUESTS}")
        else:
            # 请求成功
            self.consecutive_successes += 1
            self.consecutive_failures = 0

            # 如果连续成功次数达到阈值，尝试恢复速率
            if self.consecutive_successes >= self.success_threshold and (now - self.last_rate_adjustment_time) > 60:
                # 增加令牌生成速率，但不超过初始值
                new_rate = min(self.initial_tokens_per_minute,
                               self.token_bucket.tokens_per_minute * self.rate_limit_recovery_factor)

                if new_rate > self.token_bucket.tokens_per_minute:
                    logger.info(f"After {self.consecutive_successes} consecutive successes, increasing token generation rate: {self.token_bucket.tokens_per_minute:.0f} -> {new_rate:.0f} tokens/min")
                    print(f"✅ After {self.consecutive_successes} consecutive successes, increasing request rate: {self.token_bucket.tokens_per_minute:.0f} -> {new_rate:.0f} tokens/min")
                    self.token_bucket.tokens_per_minute = new_rate

                    # 减少最小请求间隔，但不少于初始值
                    self.MIN_REQUEST_INTERVAL = max(1.0, self.MIN_REQUEST_INTERVAL / self.rate_limit_recovery_factor)

                    # 增加最大并发请求数，但不超过初始值
                    if self.MAX_CONCURRENT_REQUESTS < 3:
                        self.MAX_CONCURRENT_REQUESTS += 1
                        self.request_semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_REQUESTS)
                        logger.info(f"Increasing maximum concurrent requests: {self.MAX_CONCURRENT_REQUESTS}")

                    self.last_rate_adjustment_time = now

    def _split_diff_content(self, diff_content: str, file_path: str = None, max_tokens_per_chunk: int = 8000) -> List[str]:
        """将大型差异内容分割成多个小块，以适应模型的上下文长度限制

        Args:
            diff_content: 差异内容
            file_path: 文件路径，用于保存diff内容
            max_tokens_per_chunk: 每个块的最大令牌数，默认为8000

        Returns:
            List[str]: 分割后的差异内容块列表
        """
        # 粗略估计令牌数
        words = diff_content.split()
        estimated_tokens = len(words) * 1.2

        # 如果启用了保存diff内容，则计算实际token数量
        if self.save_diffs and file_path:
            actual_tokens = count_tokens(diff_content, self.model_name)
            save_diff_content(file_path, diff_content, estimated_tokens, actual_tokens)

        # 如果估计的令牌数小于最大限制，直接返回原始内容
        if estimated_tokens <= max_tokens_per_chunk:
            return [diff_content]

        # 分割差异内容
        chunks = []
        lines = diff_content.split('\n')
        current_chunk = []
        current_tokens = 0

        for line in lines:
            line_tokens = len(line.split()) * 1.2

            # 如果当前块加上这一行会超过限制，则创建新块
            if current_tokens + line_tokens > max_tokens_per_chunk and current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
                current_tokens = 0

            # 如果单行就超过限制，则将其分割
            if line_tokens > max_tokens_per_chunk:
                # 将长行分割成多个小块
                words = line.split()
                sub_chunks = []
                sub_chunk = []
                sub_tokens = 0

                for word in words:
                    word_tokens = len(word) * 0.2  # 粗略估计
                    if sub_tokens + word_tokens > max_tokens_per_chunk and sub_chunk:
                        sub_chunks.append(' '.join(sub_chunk))
                        sub_chunk = []
                        sub_tokens = 0

                    sub_chunk.append(word)
                    sub_tokens += word_tokens

                if sub_chunk:
                    sub_chunks.append(' '.join(sub_chunk))

                # 将分割后的小块添加到结果中
                for sub in sub_chunks:
                    chunks.append(sub)
            else:
                # 正常添加行
                current_chunk.append(line)
                current_tokens += line_tokens

        # 添加最后一个块
        if current_chunk:
            chunks.append('\n'.join(current_chunk))

        logger.info(f"Content too large, split into {len(chunks)} chunks for evaluation")
        print(f"ℹ️ File too large, will be processed in {len(chunks)} chunks")

        # 如果启用了保存diff内容，则保存每个分割后的块
        if self.save_diffs and file_path:
            for i, chunk in enumerate(chunks):
                chunk_path = f"{file_path}.chunk{i+1}"
                chunk_tokens = count_tokens(chunk, self.model_name)
                save_diff_content(chunk_path, chunk, len(chunk.split()) * 1.2, chunk_tokens)

        return chunks

    async def _evaluate_single_diff(self, diff_content: str) -> Dict[str, Any]:
        """Evaluate a single diff with improved rate limiting."""
        # 计算文件哈希值用于缓存
        file_hash = self._calculate_file_hash(diff_content)

        # 检查缓存
        if file_hash in self.cache:
            self.cache_hits += 1
            logger.info(f"Cache hit! Retrieved evaluation result from cache (hit rate: {self.cache_hits}/{len(self.cache) + self.cache_hits})")
            return self.cache[file_hash]

        # 检查文件大小，如果过大则分块处理
        words = diff_content.split()
        estimated_tokens = len(words) * 1.2

        # 如果文件可能超过模型的上下文限制，则分块处理
        if estimated_tokens > 12000:  # 留出一些空间给系统提示和其他内容
            chunks = self._split_diff_content(diff_content)

            # 分别评估每个块
            chunk_results = []
            for i, chunk in enumerate(chunks):
                logger.info(f"Evaluating chunk {i+1}/{len(chunks)}")
                chunk_result = await self._evaluate_diff_chunk(chunk)
                chunk_results.append(chunk_result)

            # 合并结果
            merged_result = self._merge_chunk_results(chunk_results)

            # 缓存合并后的结果
            self.cache[file_hash] = merged_result
            return merged_result

        # 对于正常大小的文件，直接评估
        # 更智能地估算令牌数量 - 根据文件大小和复杂度调整系数
        complexity_factor = 1.2  # 基础系数

        # 如果文件很大，降低系数以避免过度估计
        if len(words) > 1000:
            complexity_factor = 1.0
        elif len(words) > 500:
            complexity_factor = 1.1

        estimated_tokens = len(words) * complexity_factor

        # 使用指数退避重试策略
        max_retries = 5
        retry_count = 0
        base_wait_time = 2  # 基础等待时间（秒）

        while retry_count < max_retries:
            try:
                # 获取令牌 - 使用改进的令牌桶算法
                wait_time = await self.token_bucket.get_tokens(estimated_tokens)
                if wait_time > 0:
                    logger.info(f"Rate limit: waiting {wait_time:.2f}s for token replenishment")
                    print(f"⏳ Rate limit: waiting {wait_time:.2f}s for token replenishment (current rate: {self.token_bucket.tokens_per_minute:.0f} tokens/min)")
                    # 不需要显式等待，因为令牌桶算法已经处理了等待

                # 确保请求之间有最小间隔，但使用更短的间隔
                now = time.time()
                time_since_last = now - self._last_request_time
                min_interval = max(0.5, self.MIN_REQUEST_INTERVAL - (wait_time / 2))  # 如果已经等待了一段时间，减少间隔
                if time_since_last < min_interval:
                    await asyncio.sleep(min_interval - time_since_last)

                # 发送请求到模型
                async with self.request_semaphore:
                    # 创建消息 - 使用优化的prompt
                    # 获取文件名和语言
                    file_name = "unknown"
                    language = "unknown"

                    # 尝试从diff内容中提取文件名
                    file_name_match = re.search(r'diff --git a/(.*?) b/', diff_content)
                    if file_name_match:
                        file_name = file_name_match.group(1)
                        # 猜测语言
                        language = self._guess_language(file_name)

                    # 清理代码内容，移除异常字符
                    sanitized_diff = self._sanitize_content(diff_content)

                    # 使用自定义代码评审prompt
                    review_prompt = f"""# Code Review Request

## File Information
- **File Name**: {file_name}
- **Language**: {language.lower()}

## Code to Review
```{language.lower()}
{sanitized_diff}
```

## Instructions

Please conduct a comprehensive code review following these steps:

1. **Initial Analysis**: Begin with a brief overview of the code's purpose and functionality.

2. **Detailed Evaluation**: Analyze the code across these key dimensions:

   a. **Readability** (1-10):
      - Variable and function naming clarity
      - Code organization and structure
      - Consistent formatting and indentation
      - Appropriate use of comments

   b. **Efficiency** (1-10):
      - Algorithm efficiency and complexity
      - Resource utilization (memory, CPU)
      - Optimization opportunities
      - Potential bottlenecks

   c. **Security** (1-10):
      - Input validation and sanitization
      - Authentication and authorization concerns
      - Data protection and privacy
      - Potential vulnerabilities

   d. **Structure** (1-10):
      - Modularity and separation of concerns
      - Appropriate design patterns
      - Code reusability
      - Dependency management

   e. **Error Handling** (1-10):
      - Exception handling completeness
      - Edge case coverage
      - Graceful failure mechanisms
      - Informative error messages

   f. **Documentation** (1-10):
      - Documentation completeness
      - Comment quality and relevance
      - API documentation
      - Usage examples where appropriate

   g. **Code Style** (1-10):
      - Adherence to language conventions
      - Consistency with project style
      - Readability enhancements
      - Modern language feature usage

3. **Code Change Classification**:
   - Carefully distinguish between effective and non-effective code changes
   - Non-effective changes include: whitespace adjustments, indentation fixes, comment additions, import reordering, variable/function/class renaming without behavior changes, moving code without changing logic, trivial refactoring that doesn't improve performance or maintainability, code reformatting, changing string quotes, etc.
   - Effective changes include: logic modifications, functionality additions/removals, algorithm changes, bug fixes, API changes, data structure modifications, performance optimizations, security fixes, etc.

4. **Working Hours Estimation**:
   - Estimate how many effective working hours an experienced programmer (5-10+ years) would need to complete these code changes
   - Focus primarily on effective code changes, not formatting or style changes
   - Consider code complexity, domain knowledge requirements, and context
   - Include time for understanding, implementation, testing, and integration
"""

                    # 添加语言特定的考虑因素
                    language_key = language.lower()
                    if language_key in LANGUAGE_SPECIFIC_CONSIDERATIONS:
                        review_prompt += "\n\n" + LANGUAGE_SPECIFIC_CONSIDERATIONS[language_key]

                    # 添加JSON输出指令
                    review_prompt += "\n\n" + self.json_output_instruction

                    messages = [
                        SystemMessage(content=self.system_prompt),
                        HumanMessage(content=review_prompt)
                    ]

                    # Call the model
                    response = await self.model.agenerate(messages=[messages])
                    self._last_request_time = time.time()

                    # Get response text
                    generated_text = response.generations[0][0].text

                # Parse response
                try:
                    # Extract JSON
                    json_str = self._extract_json(generated_text)
                    if not json_str:
                        logger.warning("Failed to extract JSON from response, attempting to fix")
                        json_str = self._fix_malformed_json(generated_text)

                    if not json_str:
                        logger.error("Could not extract valid JSON from the response")
                        return self._generate_default_scores("JSON parsing error. Original response: " + str(generated_text)[:500])

                    result = json.loads(json_str)

                    # Validate scores
                    scores = self._validate_scores(result)

                    # Request successful, adjust rate limits
                    self._adjust_rate_limits(is_rate_limited=False)

                    # Cache results
                    self.cache[file_hash] = scores

                    return scores

                except json.JSONDecodeError as e:
                    logger.error(f"JSON parse error: {e}")
                    logger.error(f"Raw response: {generated_text}")
                    retry_count += 1
                    if retry_count >= max_retries:
                        return self._generate_default_scores("JSON解析错误。原始响应: " + str(generated_text)[:500])
                    await asyncio.sleep(base_wait_time * (2 ** retry_count))  # 指数退避

            except Exception as e:
                error_message = str(e)
                logger.error(f"Evaluation error: {error_message}")

                # 检查是否是速率限制错误
                is_rate_limited = "rate limit" in error_message.lower() or "too many requests" in error_message.lower()

                if is_rate_limited:
                    self._adjust_rate_limits(is_rate_limited=True)
                    retry_count += 1
                    if retry_count >= max_retries:
                        return self._generate_default_scores(f"评价过程中遇到速率限制: {error_message}")
                    # 使用更长的等待时间
                    wait_time = base_wait_time * (2 ** retry_count)
                    logger.warning(f"Rate limit error, retrying in {wait_time}s (attempt {retry_count}/{max_retries})")
                    await asyncio.sleep(wait_time)
                else:
                    # 其他错误直接返回
                    return self._generate_default_scores(f"评价过程中出错: {error_message}")

        # 如果所有重试都失败
        return self._generate_default_scores("达到最大重试次数，评价失败")

    def _validate_scores(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize scores with enhanced format handling."""
        try:
            # 记录原始结果
            logger.info(f"Validating scores from result: {result}")
            print(f"DEBUG: Original LLM result: {result}")

            # 检查并处理不同格式的评分结果
            normalized_result = {}

            # 定义所有必需的字段
            required_fields = [
                "readability", "efficiency", "security", "structure",
                "error_handling", "documentation", "code_style", "overall_score", "comments", "estimated_hours"
            ]

            # 记录是否所有字段都存在
            missing_fields = [field for field in required_fields if field not in result]
            if missing_fields:
                logger.warning(f"Missing fields in result: {missing_fields}")
            else:
                logger.info("All required fields are present in the result")

            # 处理可能的不同格式
            # 格式1: {"readability": 8, "efficiency": 7, ...}
            # 格式2: {"score": {"readability": 8, "efficiency": 7, ...}}
            # 格式3: {"readability": {"score": 8}, "efficiency": {"score": 7}, ...}
            # 格式4: CODE_SUGGESTION 模板生成的格式，如 {"readability": 8.5, "efficiency_&_performance": 7.0, ...}

            # 检查是否有嵌套的评分结构
            if "score" in result and isinstance(result["score"], dict):
                # 格式2: 评分在 "score" 字段中
                score_data = result["score"]
                for field in required_fields:
                    if field in score_data:
                        normalized_result[field] = score_data[field]
                    elif field == "comments" and "evaluation" in result:
                        # 评论可能在外层的 "evaluation" 字段中
                        normalized_result["comments"] = result["evaluation"]
            else:
                # 检查格式3: 每个评分字段都是一个包含 "score" 的字典
                format3 = False
                for field in ["readability", "efficiency", "security"]:
                    if field in result and isinstance(result[field], dict) and "score" in result[field]:
                        format3 = True
                        break

                if format3:
                    # 格式3处理
                    for field in required_fields:
                        if field == "comments":
                            if "comments" in result:
                                normalized_result["comments"] = result["comments"]
                            elif "evaluation" in result:
                                normalized_result["comments"] = result["evaluation"]
                            else:
                                normalized_result["comments"] = "No evaluation comments were provided by the model. The code may require manual review."
                        elif field in result and isinstance(result[field], dict) and "score" in result[field]:
                            normalized_result[field] = result[field]["score"]
                else:
                    # 检查是否是 CODE_SUGGESTION 模板生成的格式
                    is_code_suggestion_format = False
                    if "efficiency_&_performance" in result or "final_overall_score" in result:
                        is_code_suggestion_format = True

                    if is_code_suggestion_format:
                        # 处理 CODE_SUGGESTION 模板生成的格式
                        field_mapping = {
                            "readability": "readability",
                            "efficiency_&_performance": "efficiency",
                            "efficiency": "efficiency",
                            "security": "security",
                            "structure_&_design": "structure",
                            "structure": "structure",
                            "error_handling": "error_handling",
                            "documentation_&_comments": "documentation",
                            "documentation": "documentation",
                            "code_style": "code_style",
                            "final_overall_score": "overall_score",
                            "overall_score": "overall_score",
                            "comments": "comments"
                        }

                        for source_field, target_field in field_mapping.items():
                            if source_field in result:
                                normalized_result[target_field] = result[source_field]
                    else:
                        # 格式1或其他格式，直接复制字段
                        for field in required_fields:
                            if field in result:
                                normalized_result[field] = result[field]

            # 确保所有必需字段都存在，如果缺少则使用默认值
            for field in required_fields:
                if field not in normalized_result:
                    if field == "comments":
                        # 尝试从其他可能的字段中获取评论
                        for alt_field in ["evaluation", "comment", "description", "feedback"]:
                            if alt_field in result:
                                normalized_result["comments"] = result[alt_field]
                                break
                        else:
                            normalized_result["comments"] = "No evaluation comments were provided by the model. The code may require manual review."

                # 处理嵌套的评论结构 - 无论是否在上面的循环中设置
                if field == "comments" and isinstance(normalized_result.get("comments"), dict):
                    # 如果评论是一个字典，尝试提取有用的信息并转换为字符串
                    comments_dict = normalized_result["comments"]
                    comments_str = ""

                    # 处理常见的嵌套结构
                    if "overall" in comments_dict and isinstance(comments_dict["overall"], dict) and "comment" in comments_dict["overall"]:
                        # 如果有overall评论，优先使用它
                        comments_str = comments_dict["overall"]["comment"]
                    else:
                        # 否则，尝试从各个评分字段中提取评论
                        for score_field in ["readability", "efficiency", "security", "structure", "error_handling", "documentation", "code_style"]:
                            if score_field in comments_dict and isinstance(comments_dict[score_field], dict) and "comment" in comments_dict[score_field]:
                                comments_str += f"{score_field.capitalize()}: {comments_dict[score_field]['comment']}\n"

                        # 如果没有找到任何评论，尝试直接将字典转换为字符串
                        if not comments_str:
                            try:
                                comments_str = json.dumps(comments_dict, ensure_ascii=False)
                            except:
                                comments_str = str(comments_dict)

                    normalized_result["comments"] = comments_str
                elif field == "overall_score":
                    # 检查原始结果中是否有overall_score字段
                    if "overall_score" in result:
                        # 使用原始结果中的值
                        normalized_result["overall_score"] = result["overall_score"]
                    else:
                        # 如果原始结果中没有该字段，计算其他分数的平均值
                        logger.warning("overall_score not found in original result, calculating from other scores")
                        score_fields = ["readability", "efficiency", "security", "structure",
                                      "error_handling", "documentation", "code_style"]
                        # 使用原始结果中的值计算平均分
                        available_scores = []
                        for f in score_fields:
                            if f in result:
                                try:
                                    if isinstance(result[f], str):
                                        available_scores.append(float(result[f].strip()))
                                    else:
                                        available_scores.append(float(result[f]))
                                except (ValueError, TypeError):
                                    # 如果转换失败，跳过该字段
                                    pass
                            elif f in normalized_result:
                                available_scores.append(normalized_result[f])

                        if available_scores:
                            normalized_result["overall_score"] = round(sum(available_scores) / len(available_scores), 1)
                        else:
                            logger.warning("No scores available to calculate overall_score, using default value 5.0")
                            normalized_result["overall_score"] = 5.0
                else:
                    # 检查原始结果中是否有该字段
                    if field in result:
                        # 使用原始结果中的值
                        normalized_result[field] = result[field]
                    else:
                        # 如果原始结果中没有该字段，才使用默认值5
                        logger.warning(f"Field {field} not found in original result, using default value 5")
                        normalized_result[field] = 5

            # 确保分数在有效范围内
            score_fields = ["readability", "efficiency", "security", "structure",
                          "error_handling", "documentation", "code_style"]

            for field in score_fields:
                # 确保分数是整数并在1-10范围内
                try:
                    score = normalized_result[field]
                    if isinstance(score, str):
                        score = int(score.strip())
                    elif isinstance(score, float):
                        score = round(score)

                    normalized_result[field] = max(1, min(10, score))
                except (ValueError, TypeError):
                    # 检查原始结果中是否有该字段
                    if field in result:
                        # 尝试使用原始结果中的值
                        try:
                            # 尝试转换为整数
                            if isinstance(result[field], str):
                                normalized_result[field] = int(result[field].strip())
                            elif isinstance(result[field], float):
                                normalized_result[field] = round(result[field])
                            else:
                                normalized_result[field] = result[field]
                        except (ValueError, TypeError):
                            # 如果转换失败，使用原始值
                            normalized_result[field] = result[field]
                    else:
                        # 如果原始结果中没有该字段，才使用默认值5
                        logger.warning(f"Field {field} not found in original result or could not be parsed, using default value 5")
                        normalized_result[field] = 5

            # 确保overall_score是浮点数并在1-10范围内
            try:
                overall = normalized_result["overall_score"]
                if isinstance(overall, str):
                    overall = float(overall.strip())

                normalized_result["overall_score"] = max(1.0, min(10.0, float(overall)))
            except (ValueError, TypeError):
                # 检查原始结果中是否有overall_score字段
                if "overall_score" in result:
                    # 尝试使用原始结果中的值
                    try:
                        # 尝试转换为浮点数
                        if isinstance(result["overall_score"], str):
                            normalized_result["overall_score"] = float(result["overall_score"].strip())
                        else:
                            normalized_result["overall_score"] = float(result["overall_score"])
                    except (ValueError, TypeError):
                        # 如果转换失败，使用原始值
                        normalized_result["overall_score"] = result["overall_score"]
                else:
                    # 如果原始结果中没有该字段，才使用默认值5.0
                    logger.warning("overall_score not found in original result or could not be parsed, using default value 5.0")
                    normalized_result["overall_score"] = 5.0

            # 禁用分数调整功能，保持LLM原始输出
            # 原始代码：检查所有分数是否相同，如果是，则稍微调整以增加差异性
            # scores = [normalized_result[field] for field in score_fields]
            # if len(set(scores)) <= 1:
            #     # 所有分数相同，添加一些随机变化
            #     for field in score_fields[:3]:  # 只修改前几个字段
            #         adjustment = random.choice([-1, 1])
            #         normalized_result[field] = max(1, min(10, normalized_result[field] + adjustment))

            # 确保comments字段是字符串类型
            if "comments" in normalized_result:
                if not isinstance(normalized_result["comments"], str):
                    try:
                        if isinstance(normalized_result["comments"], dict):
                            # 如果是字典，尝试提取有用的信息
                            comments_dict = normalized_result["comments"]
                            comments_str = ""

                            # 处理常见的嵌套结构
                            if "overall" in comments_dict and isinstance(comments_dict["overall"], dict) and "comment" in comments_dict["overall"]:
                                # 如果有overall评论，优先使用它
                                comments_str = comments_dict["overall"]["comment"]
                            else:
                                # 否则，尝试从各个评分字段中提取评论
                                for field in ["readability", "efficiency", "security", "structure", "error_handling", "documentation", "code_style"]:
                                    if field in comments_dict and isinstance(comments_dict[field], dict) and "comment" in comments_dict[field]:
                                        comments_str += f"{field.capitalize()}: {comments_dict[field]['comment']}\n"

                            # 如果没有找到任何评论，尝试直接将字典转换为字符串
                            if not comments_str:
                                comments_str = json.dumps(comments_dict, ensure_ascii=False)

                            normalized_result["comments"] = comments_str
                        else:
                            # 其他类型直接转换为字符串
                            normalized_result["comments"] = str(normalized_result["comments"])
                    except Exception as e:
                        logger.error(f"Error converting comments to string: {e}")
                        normalized_result["comments"] = f"Error converting evaluation comments: {str(e)}. The code may require manual review."

                # 确保评论不为空且不是简单的数字或过短的字符串
                if not normalized_result["comments"] or normalized_result["comments"].strip().isdigit() or len(normalized_result["comments"].strip()) < 10:
                    logger.warning(f"Comments field is empty, a digit, or too short: '{normalized_result['comments']}'. Attempting to extract more detailed comments from the original result.")

                    # 尝试从原始结果中提取更详细的评论
                    detailed_comments = None

                    # 检查原始结果中是否有更详细的评论
                    if "comments" in result and isinstance(result["comments"], str) and len(result["comments"]) > 10 and not result["comments"].strip().isdigit():
                        detailed_comments = result["comments"]
                    elif "evaluation" in result and isinstance(result["evaluation"], str) and len(result["evaluation"]) > 10:
                        detailed_comments = result["evaluation"]
                    elif "analysis" in result and isinstance(result["analysis"], str) and len(result["analysis"]) > 10:
                        detailed_comments = result["analysis"]

                    # 如果找到了更详细的评论，使用它
                    if detailed_comments:
                        logger.info(f"Found more detailed comments in the original result: {detailed_comments[:100]}...")
                        normalized_result["comments"] = detailed_comments
                    else:
                        normalized_result["comments"] = "No evaluation comments were provided by the model. The code may require manual review."

            # 使用from_dict方法创建CodeEvaluation实例进行最终验证
            try:
                # 记录最终结果
                logger.info(f"Final normalized result: {normalized_result}")
                print(f"DEBUG: Final normalized result: {normalized_result}")

                evaluation = CodeEvaluation.from_dict(normalized_result)
                final_result = evaluation.model_dump()

                # 记录最终模型结果
                logger.info(f"Final model result: {final_result}")
                print(f"DEBUG: Final model result: {final_result}")

                return final_result
            except Exception as e:
                logger.error(f"Error creating CodeEvaluation: {e}")
                logger.error(f"Normalized result: {normalized_result}")
                # 如果创建失败，返回一个安全的默认结果
                return self._generate_default_scores(f"验证失败: {str(e)}")
        except Exception as e:
            logger.error(f"Score validation error: {e}")
            logger.error(f"Original result: {result}")
            return self._generate_default_scores(f"分数验证错误: {str(e)}")

    def _generate_default_scores(self, error_message: str) -> Dict[str, Any]:
        """Generate default scores when evaluation fails."""
        logger.warning(f"Generating default scores due to error: {error_message[:200]}...")

        # 记录调用栈，以便了解是从哪里调用的
        import traceback
        stack_trace = traceback.format_stack()
        logger.debug(f"Default scores generated from:\n{''.join(stack_trace[-5:-1])}")

        default_scores = {
            "readability": 5,
            "efficiency": 5,
            "security": 5,
            "structure": 5,
            "error_handling": 5,
            "documentation": 5,
            "code_style": 5,
            "overall_score": 5.0,
            "effective_code_lines": 0,
            "non_effective_code_lines": 0,
            "estimated_hours": 0.0,
            "comments": f"Evaluation failed: {error_message}. The code may require manual review."
        }

        logger.info(f"Default scores generated: {default_scores}")
        return default_scores

    def _estimate_default_hours(self, additions: int, deletions: int) -> float:
        """Estimate default working hours based on additions and deletions.

        Args:
            additions: Number of added lines
            deletions: Number of deleted lines

        Returns:
            float: Estimated working hours
        """
        # Base calculation: 1 hour per 100 lines of code (additions + deletions)
        total_changes = additions + deletions

        # Base time: minimum 0.25 hours (15 minutes) for any change
        base_time = 0.25

        if total_changes <= 10:
            # Very small changes: 15-30 minutes
            return base_time
        elif total_changes <= 50:
            # Small changes: 30 minutes to 1 hour
            return base_time + (total_changes - 10) * 0.015  # ~0.6 hours for 50 lines
        elif total_changes <= 200:
            # Medium changes: 1-3 hours
            return 0.6 + (total_changes - 50) * 0.016  # ~3 hours for 200 lines
        elif total_changes <= 500:
            # Large changes: 3-6 hours
            return 3.0 + (total_changes - 200) * 0.01  # ~6 hours for 500 lines
        else:
            # Very large changes: 6+ hours
            return 6.0 + (total_changes - 500) * 0.008  # +0.8 hours per 100 lines beyond 500

    def _guess_language(self, file_path: str) -> str:
        """根据文件扩展名猜测编程语言。

        Args:
            file_path: 文件路径

        Returns:
            str: 猜测的编程语言，与 CODE_SUGGESTION 模板中的语言标准匹配
        """
        file_ext = os.path.splitext(file_path)[1].lower()

        # 文件扩展名到语言的映射，与 CODE_SUGGESTION 模板中的语言标准匹配
        ext_to_lang = {
            # Python
            '.py': 'Python',
            '.pyx': 'Python',
            '.pyi': 'Python',
            '.ipynb': 'Python',

            # JavaScript/TypeScript
            '.js': 'JavaScript',
            '.jsx': 'JavaScript',
            '.ts': 'TypeScript',
            '.tsx': 'TypeScript',
            '.mjs': 'JavaScript',

            # Java
            '.java': 'Java',
            '.jar': 'Java',
            '.class': 'Java',

            # C/C++
            '.c': 'C',
            '.cpp': 'C++',
            '.h': 'C',
            '.hpp': 'C++',

            # C#
            '.cs': 'C#',

            # Go
            '.go': 'Go',

            # Ruby
            '.rb': 'Ruby',
            '.erb': 'Ruby',

            # PHP
            '.php': 'PHP',
            '.phtml': 'PHP',

            # Swift
            '.swift': 'Swift',

            # Kotlin
            '.kt': 'Kotlin',
            '.kts': 'Kotlin',

            # Rust
            '.rs': 'Rust',

            # HTML/CSS
            '.html': 'HTML',
            '.htm': 'HTML',
            '.xhtml': 'HTML',
            '.css': 'CSS',
            '.scss': 'CSS',
            '.sass': 'CSS',
            '.less': 'CSS',

            # Shell
            '.sh': 'Shell',
            '.bash': 'Shell',
            '.zsh': 'Shell',

            # SQL
            '.sql': 'SQL',

            # 其他常见文件类型
            '.scala': 'General',
            '.hs': 'General',
            '.md': 'General',
            '.json': 'General',
            '.xml': 'General',
            '.yaml': 'General',
            '.yml': 'General',
            '.toml': 'General',
            '.ini': 'General',
            '.config': 'General',
            '.gradle': 'General',
            '.tf': 'General',
        }

        # 如果扩展名在映射中，返回对应的语言
        if file_ext in ext_to_lang:
            return ext_to_lang[file_ext]

        # 对于特殊文件名的处理
        filename = os.path.basename(file_path).lower()
        if filename == 'dockerfile':
            return 'General'
        elif filename.startswith('docker-compose'):
            return 'General'
        elif filename.startswith('makefile'):
            return 'General'
        elif filename == '.gitignore':
            return 'General'

        # 默认返回通用编程语言
        return 'General'

    async def evaluate_commits_batched(
        self,
        commits: List[CommitInfo],
        commit_file_diffs: Dict[str, Dict[str, str]],
        verbose: bool = False,
        max_files_per_batch: int = 5,
        max_tokens_per_batch: int = 12000,
    ) -> List[FileEvaluationResult]:
        """
        Evaluate multiple commits by batching multiple files into a single LLM call.

        This method significantly reduces the number of API calls by combining multiple
        file diffs into a single request, then parsing the results to separate evaluations
        for each file.

        Args:
            commits: List of commit information
            commit_file_diffs: Dictionary mapping commit hashes to file diffs
            verbose: Whether to print verbose progress information
            max_files_per_batch: Maximum number of files to include in a single batch
            max_tokens_per_batch: Maximum number of tokens to include in a single batch

        Returns:
            List of file evaluation results
        """
        # 打印统计信息
        total_files = sum(len(diffs) for diffs in commit_file_diffs.values())
        print(f"\n开始批量评估 {len(commits)} 个提交中的 {total_files} 个文件...")
        print(f"当前速率设置: {self.token_bucket.tokens_per_minute:.0f} tokens/min")
        print(f"批量评估设置: 每批最多 {max_files_per_batch} 个文件, 最多 {max_tokens_per_batch} 个令牌\n")

        # 收集所有任务
        all_tasks = []
        for commit in commits:
            if commit.hash not in commit_file_diffs:
                continue

            file_diffs = commit_file_diffs[commit.hash]
            for file_path, file_diff in file_diffs.items():
                # 估算令牌数量
                estimated_tokens = len(file_diff.split()) * 1.2
                all_tasks.append({
                    "commit": commit,
                    "file_path": file_path,
                    "file_diff": file_diff,
                    "estimated_tokens": estimated_tokens,
                    "file_size": len(file_diff)
                })

        # 按文件大小排序，小文件先处理
        all_tasks = sorted(all_tasks, key=lambda x: x["file_size"])

        # 创建批次
        batches = []
        current_batch = []
        current_batch_tokens = 0

        for task in all_tasks:
            # 如果添加这个文件会超过最大令牌数或最大文件数，则创建新批次
            if (current_batch_tokens + task["estimated_tokens"] > max_tokens_per_batch or
                len(current_batch) >= max_files_per_batch) and current_batch:
                batches.append(current_batch)
                current_batch = []
                current_batch_tokens = 0

            current_batch.append(task)
            current_batch_tokens += task["estimated_tokens"]

        # 添加最后一个批次
        if current_batch:
            batches.append(current_batch)

        print(f"将 {total_files} 个文件分成 {len(batches)} 个批次进行评估")

        # 处理每个批次
        results = []
        start_time = time.time()
        completed_files = 0

        for batch_idx, batch in enumerate(batches):
            batch_start_time = time.time()
            print(f"处理批次 {batch_idx+1}/{len(batches)}, 包含 {len(batch)} 个文件")

            try:
                # 构建批量评估请求
                batch_results = await self._evaluate_file_batch(batch)

                # 处理批量评估结果
                for task, evaluation in zip(batch, batch_results):
                    # 创建 FileEvaluationResult 对象
                    result = FileEvaluationResult(
                        file_path=task["file_path"],
                        commit_hash=task["commit"].hash,
                        commit_message=task["commit"].message,
                        date=task["commit"].date,
                        author=task["commit"].author,
                        evaluation=CodeEvaluation(**evaluation)
                    )
                    results.append(result)

                # 更新进度
                completed_files += len(batch)
                batch_time = time.time() - batch_start_time
                elapsed_time = time.time() - start_time
                avg_time_per_batch = elapsed_time / (batch_idx + 1)
                remaining_batches = len(batches) - (batch_idx + 1)
                estimated_remaining_time = avg_time_per_batch * remaining_batches

                print(f"批次 {batch_idx+1} 完成, 用时 {batch_time:.1f}s, 平均每个文件 {batch_time/len(batch):.1f}s")
                print(f"总进度: {completed_files}/{total_files} 文件 " +
                      f"({completed_files/total_files*100:.1f}%) " +
                      f"- 已用时间: {elapsed_time:.1f}s " +
                      f"- 预计剩余: {estimated_remaining_time:.1f}s")

            except Exception as e:
                logger.error(f"Error processing batch {batch_idx+1}: {str(e)}")
                print(f"处理批次 {batch_idx+1} 时出错: {str(e)}")

                # 为批次中的每个文件创建默认评价结果
                for task in batch:
                    evaluation = self._generate_default_scores(f"批量评价过程中出错: {str(e)}")

                    result = FileEvaluationResult(
                        file_path=task["file_path"],
                        commit_hash=task["commit"].hash,
                        commit_message=task["commit"].message,
                        date=task["commit"].date,
                        author=task["commit"].author,
                        evaluation=CodeEvaluation(**evaluation)
                    )
                    results.append(result)

                completed_files += len(batch)

        # 打印统计信息
        elapsed_time = time.time() - start_time
        print(f"\n批量评估完成! 总用时: {elapsed_time:.1f}s")
        print(f"平均每个批次: {elapsed_time/len(batches):.1f}s, 平均每个文件: {elapsed_time/total_files:.1f}s")
        print(f"令牌桶统计: {self.token_bucket.get_stats()}")

        return results

    async def _evaluate_file_batch(self, batch: List[Dict]) -> List[Dict]:
        """
        Evaluate a batch of files in a single LLM call.

        Args:
            batch: List of dictionaries containing file information
                Each dict should have: commit, file_path, file_diff, estimated_tokens, file_size

        Returns:
            List of evaluation results for each file in the batch
        """
        logger.info(f"Evaluating batch of {len(batch)} files")

        # Log detailed information about each file in the batch
        for i, file_info in enumerate(batch):
            file_path = file_info.get("file_path", "Unknown")
            file_size = file_info.get("file_size", 0)
            estimated_tokens = file_info.get("estimated_tokens", 0)
            logger.info(f"  Batch file {i+1}/{len(batch)}: {file_path} (size: {file_size} bytes, est. tokens: {int(estimated_tokens)})")

            # Check if required fields are present
            required_fields = ["file_path", "file_diff", "commit"]
            missing_fields = [field for field in required_fields if field not in file_info or not file_info[field]]
            if missing_fields:
                logger.warning(f"  Missing required fields for file {file_path}: {missing_fields}")

            # Check for diff_content field and convert to file_diff if needed
            if "diff_content" in file_info and "file_diff" not in file_info:
                file_info["file_diff"] = file_info["diff_content"]
                logger.info(f"  Converted diff_content to file_diff for file {file_path}")

        print(f"Processing batch with {len(batch)} files...")

        # Combine all file diffs into a single prompt
        combined_prompt = "I will provide you with multiple code diffs to evaluate. Please evaluate each file separately.\n\n"

        for i, task in enumerate(batch):
            file_path = task["file_path"]
            file_diff = task["file_diff"]

            # Determine language based on file extension
            language = self._guess_language(file_path)

            # Add file header and diff
            combined_prompt += f"\n\n## FILE {i+1}: {file_path} (Language: {language})\n\n"
            combined_prompt += f"```diff\n{file_diff}\n```\n"

        # Add evaluation instructions
        combined_prompt += "\n\n# EVALUATION INSTRUCTIONS\n"
        combined_prompt += """
For each file, provide a separate evaluation with the following scores (1-10 scale):
1. Readability: Code clarity, naming, formatting
2. Efficiency: Performance, resource usage, algorithmic efficiency
3. Security: Protection against vulnerabilities, input validation
4. Structure: Architecture, modularity, separation of concerns
5. Error Handling: Robust error handling, edge cases
6. Documentation: Comments, docstrings, self-documenting code
7. Code Style: Adherence to language conventions and best practices
8. Overall Score: Comprehensive evaluation considering all dimensions

Also estimate how many effective working hours an experienced programmer (5-10+ years) would need to implement these changes.

Format your response as a JSON array with one object per file, like this:
```json
[
  {
    "file_index": 1,
    "readability": 8,
    "efficiency": 7,
    "security": 6,
    "structure": 8,
    "error_handling": 7,
    "documentation": 6,
    "code_style": 8,
    "overall_score": 7.5,
    "estimated_hours": 2.5,
    "comments": "Detailed evaluation comments for file 1..."
  },
  {
    "file_index": 2,
    ...
  }
]
```
"""

        # Clean the combined content
        sanitized_prompt = self._sanitize_content(combined_prompt)

        # Log the batch size
        logger.info(f"Batch prompt size: {len(sanitized_prompt)} characters, approximately {len(sanitized_prompt.split()) * 1.2:.0f} tokens")

        # Create messages for the model
        messages = [HumanMessage(content=sanitized_prompt)]

        # Track retries
        max_retries = 2
        retry_count = 0

        while True:
            try:
                # Wait for token bucket if needed
                await self.token_bucket.get_tokens(len(sanitized_prompt.split()) * 1.2)

                # Get user message for logging
                user_message = messages[0].content

                # Call the model
                logger.info(f"Sending batch of {len(batch)} files to model")
                logger.info(f"Prompt size: {len(sanitized_prompt)} characters, approximately {len(sanitized_prompt.split()) * 1.2:.0f} tokens")
                print(f"Sending batch request to model (est. {len(sanitized_prompt.split()) * 1.2:.0f} tokens)...")

                start_time = time.time()
                response = await self.model.agenerate(messages=[messages])
                end_time = time.time()

                # Get response text and log details
                generated_text = response.generations[0][0].text
                logger.info(f"Batch evaluation completed in {end_time - start_time:.2f} seconds")
                logger.info(f"Response size: {len(generated_text)} characters")
                print(f"Received response from model in {end_time - start_time:.2f} seconds")

                # Log both prompt and response to the same file
                log_llm_interaction(user_message, generated_text, interaction_type="batch_evaluation",
                                  extra_info=f"Batch size: {len(batch)} files, ~{len(sanitized_prompt.split()) * 1.2:.0f} tokens")

                # Extract JSON from response
                json_str = self._extract_json(generated_text)
                if not json_str:
                    logger.warning("Failed to extract JSON from batch response, attempting to fix")
                    json_str = self._fix_malformed_json(generated_text)

                if not json_str:
                    logger.error("Could not extract valid JSON from the batch response")
                    # Create default evaluations for each file
                    return [self._generate_default_scores(f"Failed to parse batch response for file {task['file_path']}")
                            for task in batch]

                # Parse JSON
                try:
                    evaluations = json.loads(json_str)

                    # Validate that we have the correct number of evaluations
                    if not isinstance(evaluations, list):
                        logger.error(f"Expected a list of evaluations, got {type(evaluations)}")
                        evaluations = [evaluations]  # Try to convert single object to list

                    if len(evaluations) != len(batch):
                        logger.warning(f"Expected {len(batch)} evaluations, got {len(evaluations)}. Padding with defaults.")
                        # If we have fewer evaluations than files, add default evaluations
                        while len(evaluations) < len(batch):
                            missing_index = len(evaluations)
                            evaluations.append(self._generate_default_scores(
                                f"Missing evaluation for file {batch[missing_index]['file_path']}"
                            ))
                        # If we have more evaluations than files, truncate
                        evaluations = evaluations[:len(batch)]

                    # Ensure all required fields exist in each evaluation
                    for i, eval_data in enumerate(evaluations):
                        required_fields = ["readability", "efficiency", "security", "structure",
                                          "error_handling", "documentation", "code_style", "overall_score", "comments"]
                        for field in required_fields:
                            if field not in eval_data:
                                if field != "overall_score":  # overall_score can be calculated
                                    logger.warning(f"Missing field {field} in evaluation {i+1}, setting default value")
                                    eval_data[field] = 5

                        # If overall_score is not provided, calculate it
                        if "overall_score" not in eval_data or not eval_data["overall_score"]:
                            score_fields = ["readability", "efficiency", "security", "structure",
                                           "error_handling", "documentation", "code_style"]
                            scores = [eval_data.get(field, 5) for field in score_fields]
                            eval_data["overall_score"] = round(sum(scores) / len(scores), 1)

                        # If estimated_hours is not provided, add a default
                        if "estimated_hours" not in eval_data or not eval_data["estimated_hours"]:
                            eval_data["estimated_hours"] = 0.5

                    return evaluations

                except Exception as e:
                    logger.error(f"Error parsing batch evaluation: {e}", exc_info=True)
                    if retry_count < max_retries:
                        retry_count += 1
                        logger.warning(f"Retrying batch evaluation (attempt {retry_count}/{max_retries})")
                        await asyncio.sleep(2 * retry_count)  # Exponential backoff
                    else:
                        logger.error("Max retries reached, returning default evaluations")
                        return [self._generate_default_scores(f"Failed to parse batch evaluation after {max_retries} attempts")
                                for task in batch]

            except Exception as e:
                error_message = str(e)
                logger.error(f"Error during batch evaluation: {error_message}", exc_info=True)

                # Check for specific error types
                is_context_length_error = "context length" in error_message.lower() or "maximum context length" in error_message.lower()
                is_rate_limit_error = "rate limit" in error_message.lower() or "too many requests" in error_message.lower()
                is_deepseek_error = "deepseek" in error_message.lower() or "deepseek api" in error_message.lower()

                if is_context_length_error:
                    # If it's a context length error, split the batch and try again
                    logger.warning("Context length error, splitting batch")
                    if len(batch) == 1:
                        # If we can't split further, return default evaluation
                        logger.error("Cannot split batch further, returning default evaluation")
                        return [self._generate_default_scores(f"Context length error for file {task['file_path']}")
                                for task in batch]
                    else:
                        # Split the batch and evaluate each half separately
                        mid = len(batch) // 2
                        logger.info(f"Splitting batch of {len(batch)} files into batches of {mid} and {len(batch) - mid}")

                        first_half = await self._evaluate_file_batch(batch[:mid])
                        second_half = await self._evaluate_file_batch(batch[mid:])

                        return first_half + second_half

                elif is_rate_limit_error:
                    # If it's a rate limit error, wait and retry
                    if retry_count < max_retries:
                        retry_count += 1
                        wait_time = 5 * (2 ** retry_count)
                        logger.warning(f"Rate limit error, retrying in {wait_time}s (attempt {retry_count}/{max_retries})")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error("Max retries reached, returning default evaluations")
                        return [self._generate_default_scores(f"Rate limit error after {max_retries} attempts")
                                for task in batch]

                elif is_deepseek_error:
                    # For DeepSeek API errors, retry at most twice, then abandon
                    retry_count += 1
                    if retry_count >= 2:
                        logger.error("Max retries reached for DeepSeek API error, returning default evaluations")
                        return [self._generate_default_scores(f"DeepSeek API error after {retry_count} attempts")
                                for task in batch]
                    else:
                        wait_time = 5 * retry_count
                        logger.warning(f"DeepSeek API error, retrying in {wait_time}s (attempt {retry_count}/2)")
                        await asyncio.sleep(wait_time)

                else:
                    # For other errors, retry a few times
                    if retry_count < max_retries:
                        retry_count += 1
                        wait_time = 3 * retry_count
                        logger.warning(f"Error during batch evaluation, retrying in {wait_time}s (attempt {retry_count}/{max_retries})")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error("Max retries reached, returning default evaluations")
                        return [self._generate_default_scores(f"Evaluation error: {error_message}")
                                for task in batch]

    def _sanitize_content(self, content: str) -> str:
        """清理内容中的异常字符，确保内容可以安全地发送到OpenAI API。

        Args:
            content: 原始内容

        Returns:
            str: 清理后的内容
        """
        if not content:
            return ""

        try:
            # 检查是否包含Base64编码的内容
            if len(content) > 20 and content.strip().endswith('==') and all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=' for c in content.strip()):
                print(f"DEBUG: Detected possible Base64 encoded content: '{content[:20]}...'")
                return "这是一段Base64编码的内容，无法进行代码评估。"

            # 移除不可打印字符和控制字符，但保留基本空白字符（空格、换行、制表符）
            sanitized = ""
            for char in content:
                # 保留基本可打印字符和常用空白字符
                if char.isprintable() or char in [' ', '\n', '\t', '\r']:
                    sanitized += char
                else:
                    # 替换不可打印字符为空格
                    sanitized += ' '

            # 如果清理后的内容太短，返回一个提示
            if len(sanitized.strip()) < 10:
                return "代码内容太短或为空，无法进行有效评估。"

            return sanitized
        except Exception as e:
            print(f"DEBUG: Error sanitizing content: {e}")
            # 如果清理过程出错，返回一个安全的默认字符串
            return "内容清理过程中出错，无法处理。"

    def _extract_json(self, text: str) -> str:
        """从文本中提取JSON部分。

        Args:
            text: 原始文本

        Returns:
            str: 提取的JSON字符串，如果没有找到则返回空字符串
        """
        # 检查输入是否为空或None
        if not text:
            logger.warning("Empty response received from API")
            print("DEBUG: Empty response received from API")
            return ""

        # 打印原始文本的类型和长度
        logger.info(f"Response type: {type(text)}, length: {len(text)}")
        print(f"DEBUG: Response type: {type(text)}, length: {len(text)}")
        print(f"DEBUG: First 100 chars: '{text[:100]}'")

        # Log complete response for debugging
        logger.debug(f"Complete model response: {text}")

        # Check for patterns indicating unevaluable content (like Base64 encoded content)
        unevaluable_patterns = [
            r'Base64',
            r'undecodable string',
            r'ICAgIA==',
            r'cannot evaluate',
            r'cannot review this code',
            r'unable to evaluate',
            r'unable to assess the code',
            r'code is too short',
            r'code is empty',
            r'no actual code provided',
            r'cannot understand',
            r'cannot parse',
            r'cannot analyze',
            r'cannot read',
            r'cannot recognize',
            r'cannot process',
            r'invalid code',
            r'not valid code',
            r'not code',
            r'does not contain code',
            r'only contains an undecodable string'
        ]

        for pattern in unevaluable_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                print(f"DEBUG: Detected response indicating unevaluable content: '{pattern}'")
                # Extract comments if any
                comment = text[:200] if len(text) > 200 else text
                # Create a default JSON response
                default_json = {
                    "readability": 5,
                    "efficiency": 5,
                    "security": 5,
                    "structure": 5,
                    "error_handling": 5,
                    "documentation": 5,
                    "code_style": 5,
                    "overall_score": 5.0,
                    "comments": f"The code could not be evaluated due to content issues: {comment}"
                }
                return json.dumps(default_json)

        # 尝试查找JSON代码块
        json_match = re.search(r'```(?:json)?\s*({[\s\S]*?})\s*```', text)
        if json_match:
            return json_match.group(1)

        # 尝试直接查找JSON对象
        json_pattern = r'({[\s\S]*?"readability"[\s\S]*?"efficiency"[\s\S]*?"security"[\s\S]*?"structure"[\s\S]*?"error_handling"[\s\S]*?"documentation"[\s\S]*?"code_style"[\s\S]*?"overall_score"[\s\S]*?"comments"[\s\S]*?})'
        json_match = re.search(json_pattern, text)
        if json_match:
            return json_match.group(1)

        # 尝试提取 CODE_SUGGESTION 模板生成的评分部分
        scores_section = re.search(r'### SCORES:\s*\n([\s\S]*?)(?:\n\n|\Z)', text)
        if scores_section:
            scores_text = scores_section.group(1)
            scores_dict = {}

            # 提取各个评分
            for line in scores_text.split('\n'):
                # 匹配标准评分 (e.g., "- Readability: 8 /10")
                match = re.search(r'- ([\w\s&]+):\s*(\d+(\.\d+)?)\s*/10', line)
                if match:
                    key = match.group(1).strip().lower().replace(' & ', '_').replace(' ', '_')
                    value = float(match.group(2))
                    scores_dict[key] = value
                    continue

                # 匹配有效代码行数 (e.g., "- Effective Code Lines: 120")
                effective_match = re.search(r'- Effective Code Lines:\s*(\d+)', line)
                if effective_match:
                    scores_dict['effective_code_lines'] = int(effective_match.group(1))
                    continue

                # 匹配非有效代码行数 (e.g., "- Non-Effective Code Lines: 30")
                non_effective_match = re.search(r'- Non-Effective Code Lines:\s*(\d+)', line)
                if non_effective_match:
                    scores_dict['non_effective_code_lines'] = int(non_effective_match.group(1))
                    continue

                # 匹配有效添加行数 (e.g., "- Effective Additions: 100")
                effective_additions_match = re.search(r'- Effective Additions:\s*(\d+)', line)
                if effective_additions_match:
                    scores_dict['effective_additions'] = int(effective_additions_match.group(1))
                    continue

                # 匹配有效删除行数 (e.g., "- Effective Deletions: 20")
                effective_deletions_match = re.search(r'- Effective Deletions:\s*(\d+)', line)
                if effective_deletions_match:
                    scores_dict['effective_deletions'] = int(effective_deletions_match.group(1))
                    continue

                # 匹配估计工作时间 (e.g., "- Estimated Hours: 2.5")
                hours_match = re.search(r'- Estimated Hours:\s*(\d+(\.\d+)?)', line)
                if hours_match:
                    scores_dict['estimated_hours'] = float(hours_match.group(1))
                    continue

            # 提取评论部分
            analysis_match = re.search(r'## Detailed Code Analysis\s*\n([\s\S]*?)(?:\n##|\Z)', text)
            if analysis_match:
                scores_dict['comments'] = analysis_match.group(1).strip()
            else:
                # 尝试提取改进建议部分
                improvement_match = re.search(r'## Improvement Recommendations\s*\n([\s\S]*?)(?:\n##|\Z)', text)
                if improvement_match:
                    scores_dict['comments'] = improvement_match.group(1).strip()
                else:
                    # 尝试提取代码变更分析部分
                    change_analysis_match = re.search(r'## Code Change Analysis\s*\n([\s\S]*?)(?:\n##|\Z)', text)
                    if change_analysis_match:
                        scores_dict['comments'] = change_analysis_match.group(1).strip()
                    else:
                        # Try to extract any meaningful content from the response
                        overview_match = re.search(r'## Code Functionality Overview\s*\n([\s\S]*?)(?:\n##|\Z)', text)
                        if overview_match:
                            scores_dict['comments'] = overview_match.group(1).strip()
                        else:
                            # Look for any section that might contain useful information
                            for section_title in ["Summary", "Overview", "Analysis", "Evaluation", "Review", "Feedback"]:
                                section_match = re.search(f'## {section_title}\s*\n([\s\S]*?)(?:\n##|\Z)', text, re.IGNORECASE)
                                if section_match:
                                    scores_dict['comments'] = section_match.group(1).strip()
                                    break
                            else:
                                # If no sections found, use the first 500 characters of the response
                                scores_dict['comments'] = "No detailed analysis section found. Response excerpt: " + text[:500].strip()

            # 转换为 JSON 字符串
            if scores_dict and len(scores_dict) >= 8:  # 至少包含7个评分项和评论
                return json.dumps(scores_dict)

        # 尝试查找任何可能的JSON对象
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
            return text[start_idx:end_idx+1]

        # 尝试提取评分信息，即使没有完整的JSON结构
        scores_dict = {}

        # 查找评分模式，如 "Readability: 8/10" 或 "Readability score: 8"
        score_patterns = [
            r'(readability|efficiency|security|structure|error handling|documentation|code style):\s*(\d+)(?:/10)?',
            r'(readability|efficiency|security|structure|error handling|documentation|code style) score:\s*(\d+)',
        ]

        for pattern in score_patterns:
            for match in re.finditer(pattern, text.lower()):
                key = match.group(1).replace(' ', '_')
                value = int(match.group(2))
                scores_dict[key] = value

        # 如果找到了至少4个评分，认为是有效的评分信息
        if len(scores_dict) >= 4:
            # 填充缺失的评分
            for field in ["readability", "efficiency", "security", "structure", "error_handling", "documentation", "code_style"]:
                if field not in scores_dict:
                    scores_dict[field] = 5  # 默认分数

            # 计算总分
            scores_dict["overall_score"] = round(sum(scores_dict.values()) / len(scores_dict), 1)

            # 提取评论
            comment_match = re.search(r'(comments|summary|analysis|evaluation):(.*?)(?=\n\w+:|$)', text.lower(), re.DOTALL)
            if comment_match:
                scores_dict["comments"] = comment_match.group(2).strip()
            else:
                # 使用整个文本作为评论，但限制长度
                scores_dict["comments"] = text[:500] + "..." if len(text) > 500 else text

            return json.dumps(scores_dict)

        return ""

    def _fix_malformed_json(self, json_str: str) -> str:
        """尝试修复格式不正确的JSON字符串。

        Args:
            json_str: 可能格式不正确的JSON字符串

        Returns:
            str: 修复后的JSON字符串，如果无法修复则返回空字符串
        """
        # 检查输入是否为空或None
        if not json_str:
            logger.warning("Empty string passed to _fix_malformed_json")
            # 创建一个默认的JSON
            default_scores = {
                "readability": 5,
                "efficiency": 5,
                "security": 5,
                "structure": 5,
                "error_handling": 5,
                "documentation": 5,
                "code_style": 5,
                "overall_score": 5.0,
                "effective_code_lines": 0,
                "non_effective_code_lines": 0,
                "estimated_hours": 0.0,
                "comments": "No evaluation comments available. The API returned an empty response, so default scores are shown."
            }
            logger.warning("Returning default scores due to empty response")
            return json.dumps(default_scores)

        # 检查是否是错误消息而不是JSON
        error_patterns = [
            "I'm sorry",
            "there is no code",
            "please provide",
            "cannot review",
            "unable to"
        ]

        for pattern in error_patterns:
            if pattern.lower() in json_str.lower():
                logger.warning(f"API returned an error message: {json_str[:100]}...")
                print(f"DEBUG: API returned an error message: {json_str[:100]}...")
                # 创建一个默认的JSON，包含错误消息
                default_scores = {
                    "readability": 5,
                    "efficiency": 5,
                    "security": 5,
                    "structure": 5,
                    "error_handling": 5,
                    "documentation": 5,
                    "code_style": 5,
                    "overall_score": 5.0,
                    "estimated_hours": 0.0,
                    "comments": f"The evaluation could not be completed. The API returned an error message: {json_str[:200]}..."
                }
                return json.dumps(default_scores)

        original_json = json_str  # 保存原始字符串以便比较

        try:
            # 基本清理
            json_str = json_str.replace("'", '"')  # 单引号替换为双引号
            json_str = re.sub(r',\s*}', '}', json_str)  # 移除结尾的逗号
            json_str = re.sub(r',\s*]', ']', json_str)  # 移除数组结尾的逗号

            # 添加缺失的引号
            json_str = re.sub(r'([{,])\s*(\w+)\s*:', r'\1"\2":', json_str)  # 给键添加引号

            # 修复缺失的逗号
            json_str = re.sub(r'("\w+":\s*\d+|"\w+":\s*"[^"]*"|"\w+":\s*true|"\w+":\s*false|"\w+":\s*null)\s*("\w+")', r'\1,\2', json_str)

            # 尝试解析清理后的JSON
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError as e:
            error_msg = str(e)
            logger.warning(f"第一次尝试修复JSON失败: {error_msg}")

            # 如果错误与分隔符相关，尝试修复
            if "delimiter" in error_msg or "Expecting ',' delimiter" in error_msg:
                try:
                    # 获取错误位置
                    pos = e.pos
                    # 在错误位置插入逗号
                    json_str = json_str[:pos] + "," + json_str[pos:]

                    # 再次尝试
                    json.loads(json_str)
                    return json_str
                except (json.JSONDecodeError, IndexError):
                    pass

            # 尝试查找任何可能的JSON对象
            json_pattern = r'{[\s\S]*?}'
            json_matches = re.findall(json_pattern, original_json)

            if json_matches:
                # 尝试每个匹配的JSON对象
                for potential_json in json_matches:
                    try:
                        # 尝试解析
                        json.loads(potential_json)
                        return potential_json
                    except json.JSONDecodeError:
                        # 尝试基本清理
                        cleaned_json = potential_json.replace("'", '"')
                        cleaned_json = re.sub(r',\s*}', '}', cleaned_json)
                        cleaned_json = re.sub(r'([{,])\s*(\w+)\s*:', r'\1"\2":', cleaned_json)

                        try:
                            json.loads(cleaned_json)
                            return cleaned_json
                        except json.JSONDecodeError:
                            continue

            # 尝试提取分数并创建最小可用的JSON
            try:
                # 提取分数
                scores = {}
                for field in ["readability", "efficiency", "security", "structure", "error_handling", "documentation", "code_style"]:
                    # 尝试多种模式匹配
                    patterns = [
                        f'"{field}"\\s*:\\s*(\\d+)',  # "field": 8
                        f'{field}\\s*:\\s*(\\d+)',    # field: 8
                        f'{field.replace("_", " ")}\\s*:\\s*(\\d+)',  # field name: 8
                        f'{field.capitalize()}\\s*:\\s*(\\d+)',  # Field: 8
                        f'{field.replace("_", " ").title()}\\s*:\\s*(\\d+)'  # Field Name: 8
                    ]

                    for pattern in patterns:
                        match = re.search(pattern, original_json, re.IGNORECASE)
                        if match:
                            scores[field] = int(match.group(1))
                            break

                    if field not in scores:
                        scores[field] = 5  # 默认分数

                # 尝试提取总分
                overall_patterns = [
                    r'"overall_score"\s*:\s*(\d+(?:\.\d+)?)',
                    r'overall_score\s*:\s*(\d+(?:\.\d+)?)',
                    r'overall\s*:\s*(\d+(?:\.\d+)?)',
                    r'总分\s*:\s*(\d+(?:\.\d+)?)'
                ]

                for pattern in overall_patterns:
                    overall_match = re.search(pattern, original_json, re.IGNORECASE)
                    if overall_match:
                        scores["overall_score"] = float(overall_match.group(1))
                        break

                if "overall_score" not in scores:
                    # 计算总分为其他分数的平均值
                    scores["overall_score"] = round(sum(scores.values()) / len(scores), 1)

                # 尝试提取评论
                comment_patterns = [
                    r'"comments"\s*:\s*"(.*?)"',
                    r'comments\s*:\s*(.*?)(?=\n\w+:|$)',
                    r'评价\s*:\s*(.*?)(?=\n\w+:|$)',
                    r'建议\s*:\s*(.*?)(?=\n\w+:|$)'
                ]

                for pattern in comment_patterns:
                    comment_match = re.search(pattern, original_json, re.DOTALL | re.IGNORECASE)
                    if comment_match:
                        scores["comments"] = comment_match.group(1).strip()
                        break

                if "comments" not in scores:
                    # Try to extract any meaningful content from the response
                    for section_title in ["Analysis", "Evaluation", "Review", "Feedback", "Comments", "Summary", "Overview"]:
                        section_match = re.search(f'{section_title}[:\s]+([\s\S]*?)(?=\n\w+[:\s]|\Z)', original_json, re.IGNORECASE)
                        if section_match:
                            scores["comments"] = section_match.group(1).strip()
                            break
                    else:
                        # If no sections found, use the original text
                        scores["comments"] = "Extracted scores from response, but could not find detailed comments. Response excerpt: " + original_json[:300] + "..."

                # 转换为JSON字符串
                return json.dumps(scores)
            except Exception as final_e:
                logger.error(f"所有JSON修复尝试失败: {final_e}")
                logger.error(f"原始响应: {original_json[:500]}")
                print(f"无法修复JSON: {e} -> {final_e}")

                # 最后尝试：创建一个默认的JSON
                default_scores = {
                    "readability": 5,
                    "efficiency": 5,
                    "security": 5,
                    "structure": 5,
                    "error_handling": 5,
                    "documentation": 5,
                    "code_style": 5,
                    "overall_score": 5.0,
                    "effective_code_lines": 0,
                    "non_effective_code_lines": 0,
                    "estimated_hours": 0.0,
                    "comments": f"Unable to extract detailed evaluation comments. There was an error parsing the JSON response: {str(e)}. The code may require manual review."
                }
                return json.dumps(default_scores)

            return ""

    async def _evaluate_diff_chunk(self, chunk: str) -> Dict[str, Any]:
        """评估单个差异块

        Args:
            chunk: 差异内容块

        Returns:
            Dict[str, Any]: 评估结果
        """
        # 使用指数退避重试策略
        max_retries = 5
        retry_count = 0
        base_wait_time = 2  # 基础等待时间（秒）

        # 更智能地估算令牌数量
        words = chunk.split()
        complexity_factor = 1.2
        if len(words) > 1000:
            complexity_factor = 1.0
        elif len(words) > 500:
            complexity_factor = 1.1

        estimated_tokens = len(words) * complexity_factor

        while retry_count < max_retries:
            try:
                # 获取令牌
                wait_time = await self.token_bucket.get_tokens(estimated_tokens)
                if wait_time > 0:
                    logger.info(f"Rate limit: waiting {wait_time:.2f}s for token replenishment")
                    await asyncio.sleep(wait_time)

                # 确保请求之间有最小间隔
                now = time.time()
                time_since_last = now - self._last_request_time
                if time_since_last < self.MIN_REQUEST_INTERVAL:
                    await asyncio.sleep(self.MIN_REQUEST_INTERVAL - time_since_last)

                # 发送请求到模型
                async with self.request_semaphore:
                    # 创建消息 - 使用优化的prompt
                    # 获取文件名和语言
                    file_name = "unknown"
                    language = "unknown"

                    # 尝试从diff内容中提取文件名
                    file_name_match = re.search(r'diff --git a/(.*?) b/', chunk)
                    if file_name_match:
                        file_name = file_name_match.group(1)
                        # 猜测语言
                        language = self._guess_language(file_name)

                    # 使用更详细的代码评审prompt，确保模型理解任务
                    # 清理代码内容，移除异常字符
                    sanitized_chunk = self._sanitize_content(chunk)

                    review_prompt = f"""# Code Review Request

## File Information
- **File Name**: {file_name}
- **Language**: {language.lower()}

## Code to Review
```{language.lower()}
{sanitized_chunk}
```

## Instructions

Please conduct a comprehensive code review following these steps:

1. **Initial Analysis**: Begin with a brief overview of the code's purpose and functionality.

2. **Detailed Evaluation**: Analyze the code across these key dimensions:

   a. **Readability** (1-10):
      - Variable and function naming clarity
      - Code organization and structure
      - Consistent formatting and indentation
      - Appropriate use of comments

   b. **Efficiency** (1-10):
      - Algorithm efficiency and complexity
      - Resource utilization (memory, CPU)
      - Optimization opportunities
      - Potential bottlenecks

   c. **Security** (1-10):
      - Input validation and sanitization
      - Authentication and authorization concerns
      - Data protection and privacy
      - Potential vulnerabilities

   d. **Structure** (1-10):
      - Modularity and separation of concerns
      - Appropriate design patterns
      - Code reusability
      - Dependency management

   e. **Error Handling** (1-10):
      - Exception handling completeness
      - Edge case coverage
      - Graceful failure mechanisms
      - Informative error messages

   f. **Documentation** (1-10):
      - Documentation completeness
      - Comment quality and relevance
      - API documentation
      - Usage examples where appropriate

   g. **Code Style** (1-10):
      - Adherence to language conventions
      - Consistency with project style
      - Readability enhancements
      - Modern language feature usage

3. **Code Change Classification**:
   - Carefully distinguish between effective and non-effective code changes
   - Non-effective changes include: whitespace adjustments, indentation fixes, comment additions, import reordering, variable/function renaming without behavior changes, code reformatting, changing string quotes, etc.
   - Effective changes include: logic modifications, functionality additions/removals, algorithm changes, bug fixes, API changes, data structure modifications, performance optimizations, security fixes, etc.

4. **Working Hours Estimation**:
   - Estimate how many effective working hours an experienced programmer (5-10+ years) would need to complete these code changes
   - Focus primarily on effective code changes, not formatting or style changes
   - Consider code complexity, domain knowledge requirements, and context
   - Include time for understanding, implementation, testing, and integration

## Response Format

Please return your evaluation in valid JSON format with the following structure:

```json
{{
  "readability": score,
  "efficiency": score,
  "security": score,
  "structure": score,
  "error_handling": score,
  "documentation": score,
  "code_style": score,
  "overall_score": score,
  "effective_code_lines": number,
  "non_effective_code_lines": number,
  "estimated_hours": number,
  "comments": "detailed analysis with specific observations and recommendations"
}}
```

IMPORTANT: Ensure your response is valid JSON that can be parsed programmatically. If you cannot evaluate the code (e.g., incomplete or incomprehensible code), still return valid JSON with default scores of 5 and explain the reason in the comments field."""

                    # 打印完整的代码块用于调试
                    print(f"DEBUG: File name: {file_name}")
                    print(f"DEBUG: Language: {language}")
                    print(f"DEBUG: Code chunk length: {len(chunk)}")
                    print(f"DEBUG: Code chunk first 100 chars: '{chunk[:100]}'")
                    if len(chunk) < 10:
                        print(f"DEBUG: EMPTY CODE CHUNK: '{chunk}'")
                    elif len(chunk) < 100:
                        print(f"DEBUG: FULL CODE CHUNK: '{chunk}'")

                    # 如果代码块为空或太短，使用默认评分
                    if len(chunk.strip()) < 10:
                        print("DEBUG: Code chunk is too short, using default scores")
                        default_scores = {
                            "readability": 5,
                            "efficiency": 5,
                            "security": 5,
                            "structure": 5,
                            "error_handling": 5,
                            "documentation": 5,
                            "code_style": 5,
                            "overall_score": 5.0,
                            "estimated_hours": 0.25,  # Minimum 15 minutes for any change
                            "comments": f"无法评估代码，因为代码块为空或太短: '{chunk}'"
                        }
                        return default_scores

                    # 检查是否包含Base64编码的内容
                    if chunk.strip().endswith('==') and all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=' for c in chunk.strip()):
                        print(f"DEBUG: Detected possible Base64 encoded content in chunk")
                        default_scores = {
                            "readability": 5,
                            "efficiency": 5,
                            "security": 5,
                            "structure": 5,
                            "error_handling": 5,
                            "documentation": 5,
                            "code_style": 5,
                            "overall_score": 5.0,
                            "estimated_hours": 0.25,  # Minimum 15 minutes for any change
                            "comments": f"无法评估代码，因为内容可能是Base64编码: '{chunk[:50]}...'"
                        }
                        return default_scores

                    messages = [
                        SystemMessage(content=self.system_prompt),
                        HumanMessage(content=review_prompt)
                    ]

                    # 打印用户输入内容的前100个字符用于调试
                    user_message = messages[1].content if len(messages) > 1 else "No user message"
                    print(f"DEBUG: User input first 100 chars: '{user_message[:100]}...'")
                    print(f"DEBUG: User input length: {len(user_message)}")

                    # Get user message for logging
                    user_message = messages[0].content if len(messages) > 0 else "No user message"

                    # Call the model
                    response = await self.model.agenerate(messages=[messages])
                    self._last_request_time = time.time()

                    # Get response text
                    generated_text = response.generations[0][0].text

                    # Log both prompt and response to the same file
                    log_llm_interaction(user_message, generated_text, interaction_type="diff_chunk_evaluation",
                                       extra_info=f"Chunk {i+1}/{len(chunks)}")

                # 解析响应
                try:
                    # 提取JSON
                    json_str = self._extract_json(generated_text)
                    if not json_str:
                        logger.warning("Failed to extract JSON from response, attempting to fix")
                        json_str = self._fix_malformed_json(generated_text)

                    if not json_str:
                        logger.error("Could not extract valid JSON from the response")
                        return self._generate_default_scores("JSON解析错误。原始响应: " + str(generated_text)[:500])

                    result = json.loads(json_str)

                    # 验证分数
                    scores = self._validate_scores(result)

                    # 请求成功，调整速率限制
                    self._adjust_rate_limits(is_rate_limited=False)

                    return scores

                except json.JSONDecodeError as e:
                    logger.error(f"JSON parse error: {e}")
                    logger.error(f"Raw response: {generated_text}")
                    retry_count += 1
                    if retry_count >= max_retries:
                        return self._generate_default_scores("JSON解析错误。原始响应: " + str(generated_text)[:500])
                    await asyncio.sleep(base_wait_time * (2 ** retry_count))  # 指数退避

            except Exception as e:
                error_message = str(e)
                logger.error(f"Evaluation error: {error_message}")

                # 检查是否是速率限制错误
                is_rate_limited = "rate limit" in error_message.lower() or "too many requests" in error_message.lower()

                # 检查是否是上下文长度限制错误
                is_context_length_error = "context length" in error_message.lower() or "maximum context length" in error_message.lower()

                # Check if it's a DeepSeek API error
                is_deepseek_error = "deepseek" in error_message.lower() or "deepseek api" in error_message.lower()

                if is_context_length_error:
                    # If it's a context length error, try further splitting
                    logger.warning(f"Context length limit error, attempting further content splitting")
                    smaller_chunks = self._split_diff_content(chunk, max_tokens_per_chunk=4000)  # Use smaller chunk size

                    if len(smaller_chunks) > 1:
                        # If successfully split into multiple smaller chunks, evaluate each and merge results
                        sub_results = []
                        for i, sub_chunk in enumerate(smaller_chunks):
                            logger.info(f"Evaluating sub-chunk {i+1}/{len(smaller_chunks)}")
                            sub_result = await self._evaluate_diff_chunk(sub_chunk)  # Recursive call
                            sub_results.append(sub_result)

                        return self._merge_chunk_results(sub_results)
                    else:
                        # 如果无法进一步分割，返回默认评分
                        return self._generate_default_scores(f"文件过大，无法进行评估: {error_message}")
                elif is_rate_limited:
                    self._adjust_rate_limits(is_rate_limited=True)
                    retry_count += 1
                    if retry_count >= max_retries:
                        return self._generate_default_scores(f"评价过程中遇到速率限制: {error_message}")
                    # 使用更长的等待时间
                    wait_time = base_wait_time * (2 ** retry_count)
                    logger.warning(f"Rate limit error, retrying in {wait_time}s (attempt {retry_count}/{max_retries})")
                    await asyncio.sleep(wait_time)
                elif is_deepseek_error:
                    # For DeepSeek API errors, retry at most twice, then abandon
                    retry_count += 1
                    if retry_count >= 2:  # Only retry twice
                        logger.error(f"DeepSeek API error after 2 retries, abandoning evaluation: {error_message}")
                        logger.error(f"Original error: {e}")
                        logger.error(f"Last response (if any): {generated_text[:500] if generated_text else 'No response'}")

                        # Create a detailed error message
                        error_detail = f"DeepSeek API error, abandoning evaluation: {error_message}\n"
                        error_detail += f"Original error: {e}\n"
                        error_detail += f"Last response: {generated_text[:200] if generated_text else 'No response'}"

                        return self._generate_default_scores(error_detail)
                    # Use a shorter wait time
                    wait_time = 3  # Fixed 3-second wait time
                    logger.warning(f"DeepSeek API error, retrying in {wait_time}s (attempt {retry_count}/2)")
                    logger.warning(f"Error details: {error_message}")
                    await asyncio.sleep(wait_time)
                else:
                    # Return directly for other errors
                    return self._generate_default_scores(f"Error during evaluation: {error_message}")

        # If all retries fail
        return self._generate_default_scores("Maximum retry count reached, evaluation failed")

    def _merge_chunk_results(self, chunk_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge evaluation results from multiple chunks

        Args:
            chunk_results: List of evaluation results from multiple chunks

        Returns:
            Dict[str, Any]: Merged evaluation result
        """
        if not chunk_results:
            return self._generate_default_scores("No chunk evaluation results available")

        if len(chunk_results) == 1:
            return chunk_results[0]

        # 计算各个维度的平均分数
        score_fields = ["readability", "efficiency", "security", "structure",
                       "error_handling", "documentation", "code_style"]

        merged_scores = {}
        for field in score_fields:
            scores = [result.get(field, 5) for result in chunk_results]
            merged_scores[field] = round(sum(scores) / len(scores))

        # 计算总分
        overall_scores = [result.get("overall_score", 5.0) for result in chunk_results]
        merged_scores["overall_score"] = round(sum(overall_scores) / len(overall_scores), 1)

        # 计算估计工作时间 - 累加所有块的工作时间
        estimated_hours = sum(result.get("estimated_hours", 0.0) for result in chunk_results)
        # 应用一个折扣因子，因为并行处理多个块通常比顺序处理更有效率
        discount_factor = 0.8 if len(chunk_results) > 1 else 1.0
        merged_scores["estimated_hours"] = round(estimated_hours * discount_factor, 1)

        # 合并评价意见
        comments = []
        for i, result in enumerate(chunk_results):
            comment = result.get("comments", "")
            if comment:
                comments.append(f"[块 {i+1}] {comment}")

        # 如果评价意见太长，只保留前几个块的评价
        if len(comments) > 3:
            merged_comments = "\n\n".join(comments[:3]) + f"\n\n[共 {len(comments)} 个块的评价，只显示前3个块]"
        else:
            merged_comments = "\n\n".join(comments)

        merged_scores["comments"] = merged_comments or "文件分块评估，无详细评价意见。"

        return merged_scores

    async def evaluate_commit_file(
        self,
        file_path: str,
        file_diff: str,
        file_status: str = "M",
        additions: int = 0,
        deletions: int = 0,
    ) -> Dict[str, Any]:
        """
        评价单个文件的代码差异（新版本，用于commit评估）

        Args:
            file_path: 文件路径
            file_diff: 文件差异内容
            file_status: 文件状态（A=添加，M=修改，D=删除）
            additions: 添加的行数
            deletions: 删除的行数

        Returns:
            Dict[str, Any]: 文件评价结果字典，包含估计的工作时间
        """
        logger.info(f"Evaluating file: {file_path} (status: {file_status}, additions: {additions}, deletions: {deletions})")
        logger.debug(f"File diff size: {len(file_diff)} characters")
        # 如果未设置语言，根据文件扩展名猜测语言
        language = self._guess_language(file_path)
        logger.info(f"Detected language for {file_path}: {language}")

        # 清理代码内容，移除异常字符
        sanitized_diff = self._sanitize_content(file_diff)
        logger.debug(f"Sanitized diff size: {len(sanitized_diff)} characters")

        # 检查文件大小，如果过大则分块处理
        words = sanitized_diff.split()
        estimated_tokens = len(words) * 1.2
        logger.info(f"Estimated tokens for {file_path}: {estimated_tokens:.0f}")

        # 如果文件可能超过模型的上下文限制，则分块处理
        if estimated_tokens > 12000:  # 留出一些空间给系统提示和其他内容
            logger.info(f"File {file_path} is too large (estimated {estimated_tokens:.0f} tokens), will be processed in chunks")
            chunks = self._split_diff_content(sanitized_diff)
            logger.info(f"Split file into {len(chunks)} chunks")
            print(f"ℹ️ File too large, will be processed in {len(chunks)} chunks")

            # 分别评估每个块
            chunk_results = []
            for i, chunk in enumerate(chunks):
                logger.info(f"Evaluating chunk {i+1}/{len(chunks)} of {file_path}")
                logger.debug(f"Chunk {i+1} size: {len(chunk)} characters, ~{len(chunk.split())} words")
                start_time = time.time()
                chunk_result = await self._evaluate_diff_chunk(chunk)
                end_time = time.time()
                logger.info(f"Chunk {i+1} evaluation completed in {end_time - start_time:.2f} seconds")
                chunk_results.append(chunk_result)

            # 合并结果
            logger.info(f"Merging {len(chunk_results)} chunk results for {file_path}")
            merged_result = self._merge_chunk_results(chunk_results)
            logger.info(f"Merged result: overall score {merged_result.get('overall_score', 'N/A')}")

            # 添加文件信息
            result = {
                "path": file_path,
                "status": file_status,
                "additions": additions,
                "deletions": deletions,
                "readability": merged_result["readability"],
                "efficiency": merged_result["efficiency"],
                "security": merged_result["security"],
                "structure": merged_result["structure"],
                "error_handling": merged_result["error_handling"],
                "documentation": merged_result["documentation"],
                "code_style": merged_result["code_style"],
                "overall_score": merged_result["overall_score"],
                "summary": merged_result["comments"][:100] + "..." if len(merged_result["comments"]) > 100 else merged_result["comments"],
                "comments": merged_result["comments"]
            }

            return result

        # 使用 grimoire 中的 CODE_SUGGESTION 模板
        # 将模板中的占位符替换为实际值
        prompt = CODE_SUGGESTION.format(
            language=language,
            name=file_path,
            content=sanitized_diff
        )
        logger.info(f"Preparing prompt for {file_path} with language: {language}")
        logger.debug(f"Prompt size: {len(prompt)} characters")

        try:
            # 发送请求到模型
            messages = [
                HumanMessage(content=prompt)
            ]

            # 打印用户输入内容的前20个字符用于调试
            user_message = messages[0].content if len(messages) > 0 else "No user message"
            logger.debug(f"User input first 20 chars: '{user_message[:20]}...'")
            print(f"DEBUG: User input first 20 chars: '{user_message[:20]}...'")

            logger.info(f"Sending request to model for {file_path}")
            start_time = time.time()
            # Get user message for logging
            user_message = messages[0].content

            # Call the model
            response = await self.model.agenerate(messages=[messages])
            end_time = time.time()
            logger.info(f"Model response received in {end_time - start_time:.2f} seconds")

            generated_text = response.generations[0][0].text
            logger.debug(f"Response size: {len(generated_text)} characters")

            # Log both prompt and response to the same file
            log_llm_interaction(user_message, generated_text, interaction_type="file_evaluation",
                              extra_info=f"File: {file_path}")

            # 尝试提取JSON部分
            logger.info(f"Extracting JSON from response for {file_path}")
            json_str = self._extract_json(generated_text)
            if not json_str:
                logger.warning(f"Failed to extract JSON from response for {file_path}, attempting to fix")
                json_str = self._fix_malformed_json(generated_text)
                if json_str:
                    logger.info("Successfully fixed malformed JSON")
                else:
                    logger.warning("Failed to fix malformed JSON")

            if not json_str:
                logger.error(f"Could not extract valid JSON from the response for {file_path}")
                # 创建默认评价
                logger.info("Generating default scores")
                eval_data = self._generate_default_scores(f"解析错误。原始响应: {generated_text[:500]}...")
                logger.debug(f"Default scores: {eval_data}")
            else:
                # 解析JSON
                try:
                    logger.info(f"Parsing JSON for {file_path}")
                    logger.debug(f"JSON string: {json_str[:200]}...")
                    eval_data = json.loads(json_str)
                    logger.info(f"Successfully parsed JSON for {file_path}")

                    # 确保所有必要字段存在
                    required_fields = ["readability", "efficiency", "security", "structure",
                                      "error_handling", "documentation", "code_style", "overall_score", "comments"]
                    missing_fields = []
                    for field in required_fields:
                        if field not in eval_data:
                            if field != "overall_score":  # overall_score可以计算得出
                                missing_fields.append(field)
                                logger.warning(f"Missing field {field} in evaluation for {file_path}, setting default value")
                                eval_data[field] = 5

                    if missing_fields:
                        logger.warning(f"Missing fields in evaluation for {file_path}: {', '.join(missing_fields)}")

                    # 如果没有提供overall_score，计算一个
                    if "overall_score" not in eval_data or not eval_data["overall_score"]:
                        logger.info(f"Calculating overall score for {file_path}")
                        score_fields = ["readability", "efficiency", "security", "structure",
                                       "error_handling", "documentation", "code_style"]
                        scores = [eval_data.get(field, 5) for field in score_fields]
                        eval_data["overall_score"] = round(sum(scores) / len(scores), 1)
                        logger.info(f"Calculated overall score: {eval_data['overall_score']}")

                    # Log all scores
                    logger.info(f"Evaluation scores for {file_path}: " +
                               f"readability={eval_data.get('readability', 'N/A')}, " +
                               f"efficiency={eval_data.get('efficiency', 'N/A')}, " +
                               f"security={eval_data.get('security', 'N/A')}, " +
                               f"structure={eval_data.get('structure', 'N/A')}, " +
                               f"error_handling={eval_data.get('error_handling', 'N/A')}, " +
                               f"documentation={eval_data.get('documentation', 'N/A')}, " +
                               f"code_style={eval_data.get('code_style', 'N/A')}, " +
                               f"overall_score={eval_data.get('overall_score', 'N/A')}")

                except Exception as e:
                    logger.error(f"Error parsing evaluation for {file_path}: {e}", exc_info=True)
                    logger.debug(f"JSON string that caused the error: {json_str[:500]}...")
                    eval_data = self._generate_default_scores(f"解析错误。原始响应: {generated_text[:500]}...")
                    logger.debug(f"Default scores: {eval_data}")
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            eval_data = self._generate_default_scores(f"评价过程中出错: {str(e)}")

        # 确保分数不全是相同的，如果发现全是相同的评分，增加一些微小差异
        scores = [eval_data["readability"], eval_data["efficiency"], eval_data["security"],
                 eval_data["structure"], eval_data["error_handling"], eval_data["documentation"], eval_data["code_style"]]

        # 检查是否所有分数都相同，或者是否有超过75%的分数相同（例如5个3分，1个4分）
        score_counts = {}
        for score in scores:
            score_counts[score] = score_counts.get(score, 0) + 1

        most_common_score = max(score_counts, key=score_counts.get)
        most_common_count = score_counts[most_common_score]

        # 禁用分数调整功能，保持LLM原始输出
        if False:  # 原始条件: most_common_count >= 5
            logger.warning(f"Most scores are identical ({most_common_score}, count: {most_common_count}), adjusting for variety")
            print(f"检测到评分缺乏差异性 ({most_common_score}，{most_common_count}个相同)，正在调整评分使其更具差异性")

            # 根据文件扩展名和内容进行智能评分调整
            file_ext = os.path.splitext(file_path)[1].lower()

            # 设置基础分数
            base_scores = {
                "readability": most_common_score,
                "efficiency": most_common_score,
                "security": most_common_score,
                "structure": most_common_score,
                "error_handling": most_common_score,
                "documentation": most_common_score,
                "code_style": most_common_score
            }

            # 根据文件类型调整分数
            if file_ext in ['.py', '.js', '.ts', '.java', '.cs', '.cpp', '.c']:
                # 代码文件根据路径和名称进行评分调整
                if 'test' in file_path.lower():
                    # 测试文件通常:
                    # - 结构设计很重要
                    # - 但可能文档/注释稍差
                    # - 安全性通常不是重点
                    base_scores["structure"] = min(10, most_common_score + 2)
                    base_scores["documentation"] = max(1, most_common_score - 1)
                    base_scores["security"] = max(1, most_common_score - 1)
                elif 'util' in file_path.lower() or 'helper' in file_path.lower():
                    # 工具类文件通常:
                    # - 错误处理很重要
                    # - 效率可能很重要
                    base_scores["error_handling"] = min(10, most_common_score + 2)
                    base_scores["efficiency"] = min(10, most_common_score + 1)
                elif 'security' in file_path.lower() or 'auth' in file_path.lower():
                    # 安全相关文件:
                    # - 安全性很重要
                    # - 错误处理很重要
                    base_scores["security"] = min(10, most_common_score + 2)
                    base_scores["error_handling"] = min(10, most_common_score + 1)
                elif 'model' in file_path.lower() or 'schema' in file_path.lower():
                    # 模型/数据模式文件:
                    # - 代码风格很重要
                    # - 结构设计很重要
                    base_scores["code_style"] = min(10, most_common_score + 2)
                    base_scores["structure"] = min(10, most_common_score + 1)
                elif 'api' in file_path.lower() or 'endpoint' in file_path.lower():
                    # API文件:
                    # - 效率很重要
                    # - 安全性很重要
                    base_scores["efficiency"] = min(10, most_common_score + 2)
                    base_scores["security"] = min(10, most_common_score + 1)
                elif 'ui' in file_path.lower() or 'view' in file_path.lower():
                    # UI文件:
                    # - 可读性很重要
                    # - 代码风格很重要
                    base_scores["readability"] = min(10, most_common_score + 2)
                    base_scores["code_style"] = min(10, most_common_score + 1)
                else:
                    # 普通代码文件，添加随机变化，但保持合理区间
                    keys = list(base_scores.keys())
                    random.shuffle(keys)
                    # 增加两个值，减少两个值
                    for i in range(2):
                        base_scores[keys[i]] = min(10, base_scores[keys[i]] + 2)
                        base_scores[keys[i+2]] = max(1, base_scores[keys[i+2]] - 1)

            # 应用调整后的分数
            eval_data["readability"] = base_scores["readability"]
            eval_data["efficiency"] = base_scores["efficiency"]
            eval_data["security"] = base_scores["security"]
            eval_data["structure"] = base_scores["structure"]
            eval_data["error_handling"] = base_scores["error_handling"]
            eval_data["documentation"] = base_scores["documentation"]
            eval_data["code_style"] = base_scores["code_style"]

            # 重新计算平均分
            eval_data["overall_score"] = round(sum([
                eval_data["readability"],
                eval_data["efficiency"],
                eval_data["security"],
                eval_data["structure"],
                eval_data["error_handling"],
                eval_data["documentation"],
                eval_data["code_style"]
            ]) / 7, 1)

            # 记录原始分数和调整后的分数
            original_scores = {
                "readability": most_common_score,
                "efficiency": most_common_score,
                "security": most_common_score,
                "structure": most_common_score,
                "error_handling": most_common_score,
                "documentation": most_common_score,
                "code_style": most_common_score,
                "overall_score": most_common_score
            }

            adjusted_scores = {
                "readability": eval_data["readability"],
                "efficiency": eval_data["efficiency"],
                "security": eval_data["security"],
                "structure": eval_data["structure"],
                "error_handling": eval_data["error_handling"],
                "documentation": eval_data["documentation"],
                "code_style": eval_data["code_style"],
                "overall_score": eval_data["overall_score"]
            }

            logger.info(f"Original scores: {original_scores}")
            logger.info(f"Adjusted scores: {adjusted_scores}")

            # 在评论中添加分数调整说明
            adjustment_note = f"\n\n**Note**: Scores have been adjusted for differentiation. Original scores were all {most_common_score}."
            if eval_data["comments"]:
                eval_data["comments"] += adjustment_note
            else:
                eval_data["comments"] = adjustment_note

        # Calculate estimated hours if not provided
        if "estimated_hours" not in eval_data or not eval_data["estimated_hours"]:
            estimated_hours = self._estimate_default_hours(additions, deletions, file_path)
            logger.info(f"Calculated default estimated hours for {file_path}: {estimated_hours}")
        else:
            estimated_hours = eval_data["estimated_hours"]
            logger.info(f"Using model-provided estimated hours for {file_path}: {estimated_hours}")

        # 创建并返回评价结果
        result = {
            "path": file_path,
            "status": file_status,
            "additions": additions,
            "deletions": deletions,
            "readability": eval_data["readability"],
            "efficiency": eval_data["efficiency"],
            "security": eval_data["security"],
            "structure": eval_data["structure"],
            "error_handling": eval_data["error_handling"],
            "documentation": eval_data["documentation"],
            "code_style": eval_data["code_style"],
            "overall_score": eval_data["overall_score"],
            "estimated_hours": estimated_hours,
            "summary": eval_data["comments"][:100] + "..." if len(eval_data["comments"]) > 100 else eval_data["comments"],
            "comments": eval_data["comments"]
        }

        return result

    async def evaluate_file_diff(
        self,
        file_path: str,
        file_diff: str,
        commit_info: CommitInfo,
    ) -> FileEvaluationResult:
        """
        评价单个文件的代码差异

        Args:
            file_path: 文件路径
            file_diff: 文件差异内容
            commit_info: 提交信息

        Returns:
            FileEvaluationResult: 文件评价结果
        """
        # 检查文件大小，如果过大则分块处理
        words = file_diff.split()
        estimated_tokens = len(words) * 1.2

        # 如果文件可能超过模型的上下文限制，则分块处理
        if estimated_tokens > 12000:  # 留出一些空间给系统提示和其他内容
            logger.info(f"文件 {file_path} 过大（估计 {estimated_tokens:.0f} 令牌），将进行分块处理")
            print(f"ℹ️ File too large, will be processed in {len(chunks)} chunks")

            chunks = self._split_diff_content(file_diff, file_path)

            # 分别评估每个块
            chunk_results = []
            for i, chunk in enumerate(chunks):
                logger.info(f"Evaluating chunk {i+1}/{len(chunks)}")
                chunk_result = await self._evaluate_diff_chunk(chunk)
                chunk_results.append(chunk_result)

            # 合并结果
            merged_result = self._merge_chunk_results(chunk_results)

            # 创建评价结果
            return FileEvaluationResult(
                file_path=file_path,
                commit_hash=commit_info.hash,
                commit_message=commit_info.message,
                date=commit_info.date,
                author=commit_info.author,
                evaluation=CodeEvaluation(**merged_result)
            )

        # 如果未设置语言，根据文件扩展名猜测语言
        language = self._guess_language(file_path)

        # 清理代码内容，移除异常字符
        sanitized_diff = self._sanitize_content(file_diff)

        # 使用 grimoire 中的 CODE_SUGGESTION 模板
        # 将模板中的占位符替换为实际值
        prompt = CODE_SUGGESTION.format(
            language=language,
            name=file_path,
            content=sanitized_diff
        )

        # Add request for estimated working hours
        prompt += "\n\nIn addition to the code evaluation, please also estimate how many effective working hours an experienced programmer (5-10+ years) would need to complete these code changes. Include this estimate in your JSON response as 'estimated_hours'."

        try:
            # 发送请求到模型
            messages = [
                HumanMessage(content=prompt)
            ]

            # 打印用户输入内容的前20个字符用于调试
            user_message = messages[0].content if len(messages) > 0 else "No user message"
            print(f"DEBUG: User input first 20 chars: '{user_message[:20]}...'")

            # Get user message for logging
            user_message = messages[0].content

            # Call the model
            response = await self.model.agenerate(messages=[messages])
            generated_text = response.generations[0][0].text

            # Log both prompt and response to the same file
            log_llm_interaction(user_message, generated_text, interaction_type="file_evaluation",
                              extra_info=f"File: {file_path}, +{additions}/-{deletions}")

            # 尝试提取JSON部分
            json_str = self._extract_json(generated_text)
            if not json_str:
                logger.warning("Failed to extract JSON from response, attempting to fix")
                json_str = self._fix_malformed_json(generated_text)

            if not json_str:
                logger.error("Could not extract valid JSON from the response")
                # 创建默认评价
                evaluation = CodeEvaluation(
                    readability=5,
                    efficiency=5,
                    security=5,
                    structure=5,
                    error_handling=5,
                    documentation=5,
                    code_style=5,
                    overall_score=5.0,
                    comments=f"解析错误。原始响应: {generated_text[:500]}..."
                )
            else:
                # 解析JSON
                try:
                    eval_data = json.loads(json_str)

                    # 确保所有必要字段存在
                    required_fields = ["readability", "efficiency", "security", "structure",
                                      "error_handling", "documentation", "code_style", "overall_score", "comments"]
                    for field in required_fields:
                        if field not in eval_data:
                            if field != "overall_score":  # overall_score可以计算得出
                                logger.warning(f"Missing field {field} in evaluation, setting default value")
                                eval_data[field] = 5

                    # 如果没有提供overall_score，计算一个
                    if "overall_score" not in eval_data or not eval_data["overall_score"]:
                        score_fields = ["readability", "efficiency", "security", "structure",
                                       "error_handling", "documentation", "code_style"]
                        scores = [eval_data.get(field, 5) for field in score_fields]
                        eval_data["overall_score"] = round(sum(scores) / len(scores), 1)

                    # Calculate estimated hours if not provided
                    if "estimated_hours" not in eval_data or not eval_data["estimated_hours"]:
                        # Get additions and deletions from the diff
                        additions = len(re.findall(r'^\+', file_diff, re.MULTILINE))
                        deletions = len(re.findall(r'^-', file_diff, re.MULTILINE))
                        eval_data["estimated_hours"] = self._estimate_default_hours(additions, deletions, file_path)
                        logger.info(f"Calculated default estimated hours: {eval_data['estimated_hours']}")

                    # 创建评价对象
                    evaluation = CodeEvaluation(**eval_data)
                except Exception as e:
                    logger.error(f"Error parsing evaluation: {e}")
                    # Get additions and deletions from the diff
                    additions = len(re.findall(r'^\+', file_diff, re.MULTILINE))
                    deletions = len(re.findall(r'^-', file_diff, re.MULTILINE))
                    estimated_hours = self._estimate_default_hours(additions, deletions, file_path)

                    evaluation = CodeEvaluation(
                        readability=5,
                        efficiency=5,
                        security=5,
                        structure=5,
                        error_handling=5,
                        documentation=5,
                        code_style=5,
                        overall_score=5.0,
                        estimated_hours=estimated_hours,
                        comments=f"解析错误。原始响应: {generated_text[:500]}..."
                    )
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            # Get additions and deletions from the diff
            additions = len(re.findall(r'^\+', file_diff, re.MULTILINE))
            deletions = len(re.findall(r'^-', file_diff, re.MULTILINE))
            estimated_hours = self._estimate_default_hours(additions, deletions, file_path)

            evaluation = CodeEvaluation(
                readability=5,
                efficiency=5,
                security=5,
                structure=5,
                error_handling=5,
                documentation=5,
                code_style=5,
                overall_score=5.0,
                estimated_hours=estimated_hours,
                comments=f"评价过程中出错: {str(e)}"
            )

        # 确保分数不全是相同的，如果发现全是相同的评分，增加一些微小差异
        scores = [evaluation.readability, evaluation.efficiency, evaluation.security,
                 evaluation.structure, evaluation.error_handling, evaluation.documentation, evaluation.code_style]

        # 检查是否所有分数都相同，或者是否有超过75%的分数相同（例如5个3分，1个4分）
        score_counts = {}
        for score in scores:
            score_counts[score] = score_counts.get(score, 0) + 1

        most_common_score = max(score_counts, key=score_counts.get)
        most_common_count = score_counts[most_common_score]

        # 禁用分数调整功能，保持LLM原始输出
        if False:  # 原始条件: most_common_count >= 5
            logger.warning(f"Most scores are identical ({most_common_score}, count: {most_common_count}), adjusting for variety")
            print(f"检测到评分缺乏差异性 ({most_common_score}，{most_common_count}个相同)，正在调整评分使其更具差异性")

            # 根据文件扩展名和内容进行智能评分调整
            file_ext = os.path.splitext(file_path)[1].lower()

            # 设置基础分数
            base_scores = {
                "readability": most_common_score,
                "efficiency": most_common_score,
                "security": most_common_score,
                "structure": most_common_score,
                "error_handling": most_common_score,
                "documentation": most_common_score,
                "code_style": most_common_score
            }

            # 根据文件类型调整分数
            if file_ext in ['.py', '.js', '.ts', '.java', '.cs', '.cpp', '.c']:
                # 代码文件根据路径和名称进行评分调整
                if 'test' in file_path.lower():
                    # 测试文件通常:
                    # - 结构设计很重要
                    # - 但可能文档/注释稍差
                    # - 安全性通常不是重点
                    base_scores["structure"] = min(10, most_common_score + 2)
                    base_scores["documentation"] = max(1, most_common_score - 1)
                    base_scores["security"] = max(1, most_common_score - 1)
                elif 'util' in file_path.lower() or 'helper' in file_path.lower():
                    # 工具类文件通常:
                    # - 错误处理很重要
                    # - 效率可能很重要
                    base_scores["error_handling"] = min(10, most_common_score + 2)
                    base_scores["efficiency"] = min(10, most_common_score + 1)
                elif 'security' in file_path.lower() or 'auth' in file_path.lower():
                    # 安全相关文件:
                    # - 安全性很重要
                    # - 错误处理很重要
                    base_scores["security"] = min(10, most_common_score + 2)
                    base_scores["error_handling"] = min(10, most_common_score + 1)
                elif 'model' in file_path.lower() or 'schema' in file_path.lower():
                    # 模型/数据模式文件:
                    # - 代码风格很重要
                    # - 结构设计很重要
                    base_scores["code_style"] = min(10, most_common_score + 2)
                    base_scores["structure"] = min(10, most_common_score + 1)
                elif 'api' in file_path.lower() or 'endpoint' in file_path.lower():
                    # API文件:
                    # - 效率很重要
                    # - 安全性很重要
                    base_scores["efficiency"] = min(10, most_common_score + 2)
                    base_scores["security"] = min(10, most_common_score + 1)
                elif 'ui' in file_path.lower() or 'view' in file_path.lower():
                    # UI文件:
                    # - 可读性很重要
                    # - 代码风格很重要
                    base_scores["readability"] = min(10, most_common_score + 2)
                    base_scores["code_style"] = min(10, most_common_score + 1)
                else:
                    # 普通代码文件，添加随机变化，但保持合理区间
                    keys = list(base_scores.keys())
                    random.shuffle(keys)
                    # 增加两个值，减少两个值
                    for i in range(2):
                        base_scores[keys[i]] = min(10, base_scores[keys[i]] + 2)
                        base_scores[keys[i+2]] = max(1, base_scores[keys[i+2]] - 1)

            # 应用调整后的分数
            evaluation.readability = base_scores["readability"]
            evaluation.efficiency = base_scores["efficiency"]
            evaluation.security = base_scores["security"]
            evaluation.structure = base_scores["structure"]
            evaluation.error_handling = base_scores["error_handling"]
            evaluation.documentation = base_scores["documentation"]
            evaluation.code_style = base_scores["code_style"]

            # 重新计算平均分
            evaluation.overall_score = round(sum([
                evaluation.readability,
                evaluation.efficiency,
                evaluation.security,
                evaluation.structure,
                evaluation.error_handling,
                evaluation.documentation,
                evaluation.code_style
            ]) / 7, 1)

            # 记录原始分数和调整后的分数
            original_scores = {
                "readability": most_common_score,
                "efficiency": most_common_score,
                "security": most_common_score,
                "structure": most_common_score,
                "error_handling": most_common_score,
                "documentation": most_common_score,
                "code_style": most_common_score,
                "overall_score": most_common_score
            }

            adjusted_scores = {
                "readability": evaluation.readability,
                "efficiency": evaluation.efficiency,
                "security": evaluation.security,
                "structure": evaluation.structure,
                "error_handling": evaluation.error_handling,
                "documentation": evaluation.documentation,
                "code_style": evaluation.code_style,
                "overall_score": evaluation.overall_score
            }

            logger.info(f"Original scores: {original_scores}")
            logger.info(f"Adjusted scores: {adjusted_scores}")

            # 在评论中添加分数调整说明
            adjustment_note = f"\n\n**Note**: Scores have been adjusted for differentiation. Original scores were all {most_common_score}."
            if evaluation.comments:
                evaluation.comments += adjustment_note
            else:
                evaluation.comments = adjustment_note

        # 创建并返回评价结果
        return FileEvaluationResult(
            file_path=file_path,
            commit_hash=commit_info.hash,
            commit_message=commit_info.message,
            date=commit_info.date,
            author=commit_info.author,
            evaluation=evaluation
        )

    async def evaluate_commits(
        self,
        commits: List[CommitInfo],
        commit_file_diffs: Dict[str, Dict[str, str]],
        verbose: bool = False,
        use_batched_evaluation: bool = False,
        use_whole_commit_evaluation: bool = False,
        max_files_per_batch: int = 5,
        max_tokens_per_batch: int = 12000,
    ) -> List[FileEvaluationResult]:
        """
        Evaluate multiple commits with improved concurrency control.

        Args:
            commits: List of commit information
            commit_file_diffs: Dictionary mapping commit hashes to file diffs
            verbose: Whether to print verbose progress information
            use_batched_evaluation: Whether to use batched evaluation (multiple files in one LLM call)
            use_whole_commit_evaluation: Whether to evaluate the entire commit as a whole and extract individual file evaluations
            max_files_per_batch: Maximum number of files to include in a single batch
            max_tokens_per_batch: Maximum number of tokens to include in a single batch

        Returns:
            List of file evaluation results
        """
        # If whole commit evaluation is enabled, evaluate each commit as a whole
        if use_whole_commit_evaluation:
            all_results = []
            for commit in commits:
                if commit.hash not in commit_file_diffs:
                    continue

                commit_diff_dict = {}
                for file_path, file_diff in commit_file_diffs[commit.hash].items():
                    # Create a diff info dictionary for each file
                    # Estimate additions and deletions from the diff content
                    additions = len(re.findall(r'^\+', file_diff, re.MULTILINE))
                    deletions = len(re.findall(r'^-', file_diff, re.MULTILINE))
                    commit_diff_dict[file_path] = {
                        "diff": file_diff,
                        "status": "M",  # Default to modified
                        "additions": additions,
                        "deletions": deletions
                    }

                # Evaluate the commit as a whole
                whole_commit_evaluation = await self.evaluate_commit_as_whole(
                    commit.hash,
                    commit_diff_dict,
                    extract_file_evaluations=True
                )

                # Process file evaluations from whole commit evaluation
                if "file_evaluations" in whole_commit_evaluation:
                    for file_path, evaluation in whole_commit_evaluation["file_evaluations"].items():
                        # Create file evaluation result
                        file_evaluation = FileEvaluationResult(
                            file_path=file_path,
                            commit_hash=commit.hash,
                            commit_message=commit.message,
                            date=commit.date,
                            author=commit.author,
                            evaluation=CodeEvaluation(
                                readability=evaluation.get("readability", 5),
                                efficiency=evaluation.get("efficiency", 5),
                                security=evaluation.get("security", 5),
                                structure=evaluation.get("structure", 5),
                                error_handling=evaluation.get("error_handling", 5),
                                documentation=evaluation.get("documentation", 5),
                                code_style=evaluation.get("code_style", 5),
                                overall_score=evaluation.get("overall_score", 5.0),
                                estimated_hours=evaluation.get("estimated_hours", 0.5),
                                comments=evaluation.get("comments", "")
                            )
                        )
                        all_results.append(file_evaluation)

            return all_results

        # If batched evaluation is enabled, use the new method
        elif use_batched_evaluation:
            return await self.evaluate_commits_batched(
                commits,
                commit_file_diffs,
                verbose=verbose,
                max_files_per_batch=max_files_per_batch,
                max_tokens_per_batch=max_tokens_per_batch
            )
        # 打印统计信息
        total_files = sum(len(diffs) for diffs in commit_file_diffs.values())
        print(f"\n开始评估 {len(commits)} 个提交中的 {total_files} 个文件...")
        print(f"当前速率设置: {self.token_bucket.tokens_per_minute:.0f} tokens/min, 最大并发请求数: {self.MAX_CONCURRENT_REQUESTS}\n")

        # 按文件大小排序任务，先处理小文件
        evaluation_tasks = []
        task_metadata = []  # 存储每个任务的提交和文件信息

        # 收集所有任务
        for commit in commits:
            if commit.hash not in commit_file_diffs:
                continue

            file_diffs = commit_file_diffs[commit.hash]
            for file_path, file_diff in file_diffs.items():
                # 将文件大小与任务一起存储
                file_size = len(file_diff)
                evaluation_tasks.append((file_size, file_diff))
                task_metadata.append((commit, file_path))

        # 按文件大小排序，小文件先处理
        sorted_tasks = sorted(zip(evaluation_tasks, task_metadata), key=lambda x: x[0][0])
        evaluation_tasks = [task[0][1] for task in sorted_tasks]  # 只保留diff内容
        task_metadata = [task[1] for task in sorted_tasks]

        # 动态调整批处理大小
        # 根据文件数量和大小更智能地调整批大小
        if total_files > 100:
            batch_size = 1  # 很多文件时，使用串行处理
        elif total_files > 50:
            batch_size = 2  # 较多文件时，使用小批大小
        elif total_files > 20:
            batch_size = max(2, self.MAX_CONCURRENT_REQUESTS - 1)  # 中等数量文件
        else:
            batch_size = self.MAX_CONCURRENT_REQUESTS  # 少量文件时使用完整并发

        # 检查文件大小，如果有大文件，进一步减小批大小
        large_files = sum(1 for task in evaluation_tasks if len(task.split()) > 5000)
        if large_files > 10 and batch_size > 1:
            batch_size = max(1, batch_size - 1)
            print(f"检测到 {large_files} 个大文件，减小批大小为 {batch_size}")

        print(f"使用批大小: {batch_size}")

        results = []
        start_time = time.time()
        completed_tasks = 0

        for i in range(0, len(evaluation_tasks), batch_size):
            # 创建批处理任务
            batch_tasks = []
            for diff in evaluation_tasks[i:i + batch_size]:
                batch_tasks.append(self._evaluate_single_diff(diff))

            # 使用 gather 并发执行任务，但设置 return_exceptions=True 以便在一个任务失败时继续处理其他任务
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            # 创建 FileEvaluationResult 对象
            for j, eval_result in enumerate(batch_results):
                task_idx = i + j
                if task_idx >= len(task_metadata):
                    break

                commit, file_path = task_metadata[task_idx]

                # 检查是否发生异常
                if isinstance(eval_result, Exception):
                    logger.error(f"Error evaluating file {file_path}: {str(eval_result)}")
                    print(f"⚠️ Error evaluating file {file_path}: {str(eval_result)}")

                    # 创建默认评估结果
                    default_scores = self._generate_default_scores(f"评估失败: {str(eval_result)}")
                    results.append(
                        FileEvaluationResult(
                            file_path=file_path,
                            commit_hash=commit.hash,
                            commit_message=commit.message,
                            date=commit.date,
                            author=commit.author,
                            evaluation=CodeEvaluation(**default_scores)
                        )
                    )
                else:
                    # 正常处理评估结果
                    try:
                        results.append(
                            FileEvaluationResult(
                                file_path=file_path,
                                commit_hash=commit.hash,
                                commit_message=commit.message,
                                date=commit.date,
                                author=commit.author,
                                evaluation=CodeEvaluation(**eval_result)
                            )
                        )
                    except Exception as e:
                        logger.error(f"Error creating evaluation result object: {str(e)}\nEvaluation result: {eval_result}")
                        print(f"⚠️ 创建评估结果对象时出错: {str(e)}")

                        # 创建默认评估结果
                        default_scores = self._generate_default_scores(f"处理评估结果时出错: {str(e)}")
                        results.append(
                            FileEvaluationResult(
                                file_path=file_path,
                                commit_hash=commit.hash,
                                commit_message=commit.message,
                                date=commit.date,
                                author=commit.author,
                                evaluation=CodeEvaluation(**default_scores)
                            )
                        )

                # 更新进度
                completed_tasks += 1
                elapsed_time = time.time() - start_time
                estimated_total_time = (elapsed_time / completed_tasks) * total_files
                remaining_time = estimated_total_time - elapsed_time

                # 每完成 5 个任务或每个批次结束时显示进度
                if completed_tasks % 5 == 0 or j == len(batch_results) - 1:
                    print(f"进度: {completed_tasks}/{total_files} 文件 ({completed_tasks/total_files*100:.1f}%) - 预计剩余时间: {remaining_time/60:.1f} 分钟")

            # 批次之间添加自适应延迟
            if i + batch_size < len(evaluation_tasks):
                # 根据文件大小、数量和当前令牌桶状态调整延迟

                # 获取令牌桶统计信息
                token_stats = self.token_bucket.get_stats()
                tokens_available = token_stats.get("current_tokens", 0)
                tokens_per_minute = token_stats.get("tokens_per_minute", 6000)

                # 计算下一批文件的估计令牌数
                next_batch_start = min(i + batch_size, len(evaluation_tasks))
                next_batch_end = min(next_batch_start + batch_size, len(evaluation_tasks))
                next_batch_tokens = sum(len(task.split()) * 1.2 for task in evaluation_tasks[next_batch_start:next_batch_end])

                # 如果令牌桶中的令牌不足以处理下一批，计算需要等待的时间
                if tokens_available < next_batch_tokens:
                    tokens_needed = next_batch_tokens - tokens_available
                    wait_time = (tokens_needed * 60.0 / tokens_per_minute) * 0.8  # 等待时间稍微减少一点，因为令牌桶会自动处理等待

                    # 设置最小和最大等待时间
                    delay = max(0.5, min(5.0, wait_time))

                    if verbose:
                        print(f"令牌桶状态: {tokens_available:.0f}/{tokens_per_minute:.0f} tokens, 下一批需要: {next_batch_tokens:.0f} tokens, 等待: {delay:.1f}s")
                else:
                    # 如果有足够的令牌，使用最小延迟
                    delay = 0.5

                # 根据文件数量调整基础延迟
                if total_files > 100:
                    delay = max(delay, 3.0)  # 大量文件时使用更长的延迟
                elif total_files > 50:
                    delay = max(delay, 2.0)
                elif total_files > 20:
                    delay = max(delay, 1.0)

                # 如果最近有速率限制错误，增加延迟
                if self.rate_limit_errors > 0:
                    delay *= (1 + min(3, self.rate_limit_errors) * 0.5)  # 最多增加 3 倍

                # 最终限制延迟范围
                delay = min(10.0, max(0.5, delay))  # 确保延迟在 0.5-10 秒之间

                if verbose:
                    print(f"批次间延迟: {delay:.1f}s")

                await asyncio.sleep(delay)

        # 打印统计信息
        total_time = time.time() - start_time
        print(f"\n评估完成! 总耗时: {total_time/60:.1f} 分钟")
        print(f"缓存命中率: {self.cache_hits}/{len(self.cache) + self.cache_hits} ({self.cache_hits/(len(self.cache) + self.cache_hits)*100 if len(self.cache) + self.cache_hits > 0 else 0:.1f}%)")
        print(f"令牌桶统计: {self.token_bucket.get_stats()}")

        return results

    async def evaluate_commit_as_whole(
        self,
        commit_hash: str,
        commit_diff: Dict[str, Dict[str, Any]],
        extract_file_evaluations: bool = False,
    ) -> Dict[str, Any]:
        """Evaluate all diffs in a commit together as a whole.

        This method combines all file diffs into a single evaluation to get a holistic view
        of the commit and estimate the effective working hours needed.

        Args:
            commit_hash: The hash of the commit being evaluated
            commit_diff: Dictionary mapping file paths to their diffs and statistics
            extract_file_evaluations: Whether to extract individual file evaluations from the whole commit evaluation

        Returns:
            Dictionary containing evaluation results including estimated working hours and optionally individual file evaluations
        """
        logger.info(f"Starting whole-commit evaluation for {commit_hash}")

        # Combine all diffs into a single string with file headers
        combined_diff = ""
        total_additions = 0
        total_deletions = 0

        for file_path, diff_info in commit_diff.items():
            file_diff = diff_info["diff"]
            status = diff_info["status"]
            additions = diff_info.get("additions", 0)
            deletions = diff_info.get("deletions", 0)

            total_additions += additions
            total_deletions += deletions

            # Add file header
            combined_diff += f"\n\n### File: {file_path} (Status: {status}, +{additions}, -{deletions})\n\n"
            combined_diff += file_diff

        logger.info(f"Combined {len(commit_diff)} files into a single evaluation")
        logger.debug(f"Combined diff size: {len(combined_diff)} characters")

        # Clean the combined diff content
        sanitized_diff = self._sanitize_content(combined_diff)

        # Check if the combined diff is too large
        words = sanitized_diff.split()
        estimated_tokens = len(words) * 1.2
        logger.info(f"Estimated tokens for combined diff: {estimated_tokens:.0f}")

        # Create a prompt for evaluating the entire commit
        language = "multiple"  # Since we're evaluating multiple files

        # Create a prompt that specifically asks for working hours estimation and distinguishes effective changes
        prompt = f"""Act as a senior code reviewer with 10+ years of experience. I will provide you with a complete diff of a commit that includes multiple files.

Please analyze the entire commit as a whole and provide:

1. A comprehensive evaluation of the code changes
2. Carefully distinguish between effective code changes and non-effective changes:
   - Non-effective changes (should NOT count toward working hours):
     * Whitespace adjustments (spaces, tabs, line breaks)
     * Indentation fixes without logic changes
     * Comment additions or modifications without code changes
     * Import reordering or reorganization
     * Variable/function renaming without behavior changes
     * Code reformatting (e.g., line wrapping, bracket placement)
     * Changing string quotes (e.g., single to double quotes)
     * Adding/removing trailing commas
     * Changing code style to match linter rules

   - Effective changes (SHOULD count toward working hours):
     * Logic modifications that alter program behavior
     * Functionality additions or removals
     * Algorithm changes or optimizations
     * Bug fixes that correct actual issues
     * API changes (parameters, return types, etc.)
     * Data structure modifications
     * Performance optimizations
     * Security vulnerability fixes
     * Error handling improvements

3. Count the number of effective code lines changed and non-effective code lines changed:
   - For each file, analyze line by line to determine if changes are effective or non-effective
   - Count both added and removed lines, but categorize them correctly
   - For mixed changes (both effective and non-effective in same line), count as effective
   - Provide a breakdown of effective vs. non-effective changes by file

4. Estimate how many effective working hours an experienced programmer (5-10+ years) would need:
   - Base your estimate primarily on effective code changes, not total changes
   - For purely non-effective changes (only formatting/style):
     * 0.1-0.2 hours for small files
     * 0.3-0.5 hours for large files with extensive formatting
   - For effective changes, consider:
     * Complexity of the logic being modified
     * Domain knowledge required to understand the code
     * Testing requirements for the changes
     * Integration complexity with other components
   - Be realistic - experienced programmers work efficiently but still need time to:
     * Understand existing code
     * Design appropriate solutions
     * Implement changes carefully
     * Test and verify correctness
5. Scores for the following aspects (1-10 scale):
   - Readability
   - Efficiency
   - Security
   - Structure
   - Error Handling
   - Documentation
   - Code Style
   - Overall Score

6. Evaluate each file individually with scores for the same aspects (1-10 scale)

Here's the complete diff for commit {commit_hash}:

```
{sanitized_diff}
```

Please format your response as JSON with the following fields:
- readability: (score 1-10)
- efficiency: (score 1-10)
- security: (score 1-10)
- structure: (score 1-10)
- error_handling: (score 1-10)
- documentation: (score 1-10)
- code_style: (score 1-10)
- overall_score: (score 1-10)
- effective_code_lines: (number of lines with actual logic/functionality changes)
- non_effective_code_lines: (number of lines with formatting, whitespace, comment changes)
- estimated_hours: (number of hours based primarily on effective changes)
- comments: (your detailed analysis including breakdown of effective vs non-effective changes)
- file_evaluations: (an object with file paths as keys, each containing individual evaluations with the same scoring fields)
"""

        logger.info("Preparing to evaluate combined diff")
        logger.debug(f"Prompt size: {len(prompt)} characters")

        try:
            # Send request to model
            messages = [HumanMessage(content=prompt)]

            logger.info("Sending request to model for combined diff evaluation")
            start_time = time.time()
            # Get user message for logging
            user_message = messages[0].content

            # Call the model
            response = await self.model.agenerate(messages=[messages])
            end_time = time.time()
            logger.info(f"Model response received in {end_time - start_time:.2f} seconds")

            generated_text = response.generations[0][0].text

            # Log both prompt and response to the same file with whole-commit info
            commit_info = {
                "commit_hash": commit_hash,
                "files_count": len(commit_diff),
                "total_additions": total_additions,
                "total_deletions": total_deletions,
                "estimated_tokens": estimated_tokens
            }
            log_llm_interaction(user_message, generated_text, interaction_type="whole_commit_evaluation",
                               extra_info=f"Commit: {commit_hash}, Files: {len(commit_diff)}, +{total_additions}/-{total_deletions}, ~{estimated_tokens:.0f} tokens")
            logger.debug(f"Response size: {len(generated_text)} characters")

            # Extract JSON from response
            logger.info("Extracting JSON from response")
            json_str = self._extract_json(generated_text)
            if not json_str:
                logger.warning("Failed to extract JSON from response, attempting to fix")
                json_str = self._fix_malformed_json(generated_text)

            if not json_str:
                logger.error("Could not extract valid JSON from the response")
                # Create default evaluation
                eval_data = self._generate_default_scores("Failed to parse response")
                eval_data["estimated_hours"] = self._estimate_default_hours(total_additions, total_deletions)
            else:
                # Parse JSON
                try:
                    eval_data = json.loads(json_str)

                    # Ensure all necessary fields exist
                    required_fields = ["readability", "efficiency", "security", "structure",
                                      "error_handling", "documentation", "code_style", "overall_score", "comments"]
                    for field in required_fields:
                        if field not in eval_data:
                            if field != "overall_score":  # overall_score can be calculated
                                logger.warning(f"Missing field {field} in evaluation, setting default value")
                                eval_data[field] = 5

                    # Add effective and non-effective code lines if not present
                    if "effective_code_lines" not in eval_data:
                        # Estimate based on additions and deletions
                        logger.warning("Missing effective_code_lines in evaluation, estimating")
                        # Assume 60% of changes are effective by default
                        eval_data["effective_code_lines"] = int(total_additions * 0.6) + int(total_deletions * 0.6)
                        logger.info(f"Estimated effective_code_lines: {eval_data['effective_code_lines']}")
                    else:
                        logger.info(f"Using LLM evaluated effective_code_lines: {eval_data['effective_code_lines']}")

                    if "non_effective_code_lines" not in eval_data:
                        logger.warning("Missing non_effective_code_lines in evaluation, estimating")
                        # Assume 40% of changes are non-effective by default
                        eval_data["non_effective_code_lines"] = int(total_additions * 0.4) + int(total_deletions * 0.4)
                        logger.info(f"Estimated non_effective_code_lines: {eval_data['non_effective_code_lines']}")
                    else:
                        logger.info(f"Using LLM evaluated non_effective_code_lines: {eval_data['non_effective_code_lines']}")

                    # Add effective additions and deletions if not present
                    if "effective_additions" not in eval_data:
                        logger.warning("Missing effective_additions in evaluation, estimating")
                        # Assume 70% of effective lines are additions
                        eval_data["effective_additions"] = int(eval_data["effective_code_lines"] * 0.7)

                    if "effective_deletions" not in eval_data:
                        logger.warning("Missing effective_deletions in evaluation, estimating")
                        # Assume 30% of effective lines are deletions
                        eval_data["effective_deletions"] = int(eval_data["effective_code_lines"] * 0.3)

                    # If overall_score is not provided, calculate it
                    if "overall_score" not in eval_data or not eval_data["overall_score"]:
                        score_fields = ["readability", "efficiency", "security", "structure",
                                       "error_handling", "documentation", "code_style"]
                        scores = [eval_data.get(field, 5) for field in score_fields]
                        eval_data["overall_score"] = round(sum(scores) / len(scores), 1)

                    # If estimated_hours is not provided, calculate a default with improved logic
                    if "estimated_hours" not in eval_data or not eval_data["estimated_hours"]:
                        logger.warning("Missing estimated_hours in evaluation, calculating default")

                        # Calculate effective code lines if available
                        effective_lines = eval_data.get("effective_code_lines", int(total_additions * 0.6) + int(total_deletions * 0.6))
                        non_effective_lines = eval_data.get("non_effective_code_lines", int(total_additions * 0.4) + int(total_deletions * 0.4))

                        # Calculate hours based on effective code lines with higher weight
                        effective_hours = self._estimate_default_hours(
                            int(effective_lines * 0.7),  # Consider 70% of effective lines as additions
                            int(effective_lines * 0.3)   # Consider 30% of effective lines as deletions
                        )

                        # Calculate hours for non-effective code with lower weight
                        non_effective_hours = self._estimate_default_hours(
                            int(non_effective_lines * 0.7),
                            int(non_effective_lines * 0.3)
                        ) * 0.3  # Apply 30% weight to non-effective changes

                        # Calculate total hours
                        total_hours = effective_hours + non_effective_hours

                        # Apply commit complexity factor based on number of files
                        file_count = len(commit_diff)
                        if file_count > 10:
                            # For large commits with many files, add integration complexity
                            complexity_factor = 1.2 + (min(file_count, 50) - 10) * 0.01  # Max +40% for 50+ files
                        elif file_count > 5:
                            # Medium complexity for 6-10 files
                            complexity_factor = 1.1
                        elif file_count > 1:
                            # Slight complexity for 2-5 files
                            complexity_factor = 1.05
                        else:
                            # No additional complexity for single file
                            complexity_factor = 1.0

                        # Apply complexity factor
                        eval_data["estimated_hours"] = round(total_hours * complexity_factor * 10) / 10

                    # If file_evaluations is not provided, create default evaluations for each file
                    if "file_evaluations" not in eval_data and extract_file_evaluations:
                        logger.warning("Missing file_evaluations in evaluation, creating defaults")
                        eval_data["file_evaluations"] = {}

                        # Create default evaluations for each file
                        for file_path, diff_info in commit_diff.items():
                            additions = diff_info.get("additions", 0)
                            deletions = diff_info.get("deletions", 0)

                            # Use the overall scores as default for each file
                            eval_data["file_evaluations"][file_path] = {
                                "readability": eval_data.get("readability", 5),
                                "efficiency": eval_data.get("efficiency", 5),
                                "security": eval_data.get("security", 5),
                                "structure": eval_data.get("structure", 5),
                                "error_handling": eval_data.get("error_handling", 5),
                                "documentation": eval_data.get("documentation", 5),
                                "code_style": eval_data.get("code_style", 5),
                                "overall_score": eval_data.get("overall_score", 5),
                                "estimated_hours": self._estimate_default_hours(additions, deletions, file_path),
                                "comments": "Generated from whole commit evaluation."
                            }

                    # Log all scores and code line counts
                    logger.info(f"Whole commit evaluation scores: " +
                               f"readability={eval_data.get('readability', 'N/A')}, " +
                               f"efficiency={eval_data.get('efficiency', 'N/A')}, " +
                               f"security={eval_data.get('security', 'N/A')}, " +
                               f"structure={eval_data.get('structure', 'N/A')}, " +
                               f"error_handling={eval_data.get('error_handling', 'N/A')}, " +
                               f"documentation={eval_data.get('documentation', 'N/A')}, " +
                               f"code_style={eval_data.get('code_style', 'N/A')}, " +
                               f"overall_score={eval_data.get('overall_score', 'N/A')}, " +
                               f"effective_code_lines={eval_data.get('effective_code_lines', 'N/A')}, " +
                               f"non_effective_code_lines={eval_data.get('non_effective_code_lines', 'N/A')}, " +
                               f"estimated_hours={eval_data.get('estimated_hours', 'N/A')}")

                    # Log file evaluations if available
                    if "file_evaluations" in eval_data:
                        logger.info(f"Extracted evaluations for {len(eval_data['file_evaluations'])} files")

                except Exception as e:
                    logger.error(f"Error parsing evaluation: {e}", exc_info=True)
                    eval_data = self._generate_default_scores(f"解析错误。原始响应: {generated_text[:500]}...")
                    # Calculate estimated hours with improved logic for whole commit
                    total_hours = 0
                    for file_path, diff_info in commit_diff.items():
                        additions = diff_info.get("additions", 0)
                        deletions = diff_info.get("deletions", 0)
                        file_hours = self._estimate_default_hours(additions, deletions, file_path)
                        total_hours += file_hours

                    # Apply integration factor for multiple files
                    file_count = len(commit_diff)
                    if file_count > 1:
                        integration_factor = 1.0 + min(0.3, (file_count - 1) * 0.05)  # Max +30% for 7+ files
                        total_hours *= integration_factor

                    eval_data["estimated_hours"] = round(total_hours * 10) / 10

        except Exception as e:
            logger.error(f"Error during evaluation: {e}", exc_info=True)
            eval_data = self._generate_default_scores(f"评价过程中出错: {str(e)}")
            # Calculate estimated hours with improved logic for whole commit
            total_hours = 0
            for file_path, diff_info in commit_diff.items():
                additions = diff_info.get("additions", 0)
                deletions = diff_info.get("deletions", 0)
                file_hours = self._estimate_default_hours(additions, deletions, file_path)
                total_hours += file_hours

            # Apply integration factor for multiple files
            file_count = len(commit_diff)
            if file_count > 1:
                integration_factor = 1.0 + min(0.3, (file_count - 1) * 0.05)  # Max +30% for 7+ files
                total_hours *= integration_factor

            eval_data["estimated_hours"] = round(total_hours * 10) / 10

        return eval_data

    def _estimate_default_hours(self, additions: int, deletions: int, file_path: str = None) -> float:
        """Estimate default working hours based on additions and deletions.

        This is a fallback method when the model doesn't provide an estimate.
        Uses a more granular approach with a minimum of 0.1 hours (6 minutes) for very small changes.

        Args:
            additions: Number of lines added
            deletions: Number of lines deleted
            file_path: Optional file path to consider file type in estimation

        Returns:
            float: Estimated working hours
        """
        # Calculate total changes
        total_changes = additions + deletions

        # Base calculation with more granular approach
        if total_changes <= 5:
            # Very small changes (1-5 lines): 0.1 hours (6 minutes)
            base_hours = 0.1
        elif total_changes <= 10:
            # Small changes (6-10 lines): 0.2 hours (12 minutes)
            base_hours = 0.2
        elif total_changes <= 20:
            # Medium-small changes (11-20 lines): 0.3 hours (18 minutes)
            base_hours = 0.3
        elif total_changes <= 50:
            # Medium changes (21-50 lines): 0.5-1 hour
            base_hours = 0.5 + (total_changes - 20) * 0.016  # ~1 hour for 50 lines
        elif total_changes <= 100:
            # Medium-large changes (51-100 lines): 1-2 hours
            base_hours = 1.0 + (total_changes - 50) * 0.02  # ~2 hours for 100 lines
        elif total_changes <= 200:
            # Large changes (101-200 lines): 2-3.5 hours
            base_hours = 2.0 + (total_changes - 100) * 0.015  # ~3.5 hours for 200 lines
        elif total_changes <= 500:
            # Very large changes (201-500 lines): 3.5-8 hours
            base_hours = 3.5 + (total_changes - 200) * 0.015  # ~8 hours for 500 lines
        else:
            # Massive changes (500+ lines): 8+ hours
            base_hours = 8.0 + (total_changes - 500) * 0.01  # +1 hour per 100 lines beyond 500

        # Apply complexity factor based on file type if file_path is provided
        complexity_factor = 1.0
        if file_path:
            file_ext = os.path.splitext(file_path)[1].lower() if file_path else ""

            # Higher complexity for certain file types
            if file_ext in ['.c', '.cpp', '.h', '.hpp']:
                complexity_factor = 1.3  # C/C++ tends to be more complex
            elif file_ext in ['.java', '.scala']:
                complexity_factor = 1.2  # Java/Scala slightly more complex
            elif file_ext in ['.py', '.js', '.ts']:
                complexity_factor = 1.0  # Python/JavaScript/TypeScript standard complexity
            elif file_ext in ['.html', '.css', '.md', '.txt', '.json', '.yaml', '.yml']:
                complexity_factor = 0.8  # Markup/config files tend to be simpler

            # Consider file path indicators of complexity
            if 'test' in file_path.lower() or 'spec' in file_path.lower():
                complexity_factor *= 0.9  # Test files often simpler to modify
            if 'core' in file_path.lower() or 'engine' in file_path.lower():
                complexity_factor *= 1.2  # Core functionality often more complex
            if 'util' in file_path.lower() or 'helper' in file_path.lower():
                complexity_factor *= 0.9  # Utility functions often simpler

        # Apply the complexity factor
        estimated_hours = base_hours * complexity_factor

        # Round to 1 decimal place with proper precision
        estimated_hours = round(estimated_hours * 10) / 10

        # Ensure minimum of 0.1 hours and maximum of 40 hours (1 week)
        return max(0.1, min(40, estimated_hours))

    async def evaluate_commit(
        self,
        commit_hash: str,
        commit_diff: Dict[str, Dict[str, Any]],
        use_batched_evaluation: bool = False,
        max_files_per_batch: int = 5,
        max_tokens_per_batch: int = 12000,
        use_whole_commit_evaluation: bool = False,
    ) -> Dict[str, Any]:
        """Evaluate a specific commit's changes.

        Args:
            commit_hash: The hash of the commit being evaluated
            commit_diff: Dictionary mapping file paths to their diffs and statistics
            use_batched_evaluation: Whether to use batched evaluation (multiple files in one LLM call)
            max_files_per_batch: Maximum number of files to include in a single batch
            max_tokens_per_batch: Maximum number of tokens to include in a single batch
            use_whole_commit_evaluation: Whether to evaluate the entire commit as a whole and extract individual file evaluations

        Returns:
            Dictionary containing evaluation results
        """
        logger.info(f"Starting evaluation for commit {commit_hash}")
        logger.info(f"Found {len(commit_diff)} files to evaluate")

        # Log file statistics
        total_additions = sum(diff.get("additions", 0) for diff in commit_diff.values())
        total_deletions = sum(diff.get("deletions", 0) for diff in commit_diff.values())
        logger.info(f"Commit statistics: {len(commit_diff)} files, {total_additions} additions, {total_deletions} deletions")

        # Initialize evaluation results
        evaluation_results = {
            "commit_hash": commit_hash,
            "files": [],
            "summary": "",
            "statistics": {
                "total_files": len(commit_diff),
                "total_additions": total_additions,
                "total_deletions": total_deletions,
                "total_effective_lines": 0,  # Will be updated with LLM evaluation results later
            }
        }
        logger.debug(f"Initialized evaluation results structure for commit {commit_hash}")

        # Evaluate each file
        logger.info(f"Starting file evaluation for commit {commit_hash}")

        # Check if we should use whole commit evaluation
        if use_whole_commit_evaluation:
            logger.info(f"Using whole commit evaluation")
            print(f"Using whole commit evaluation (all files evaluated together)")

            # Evaluate the entire commit as a whole and extract individual file evaluations
            whole_commit_evaluation = await self.evaluate_commit_as_whole(commit_hash, commit_diff, extract_file_evaluations=True)

            # Process file evaluations from whole commit evaluation
            if "file_evaluations" in whole_commit_evaluation:
                for file_path, evaluation in whole_commit_evaluation["file_evaluations"].items():
                    # Get file info
                    diff_info = commit_diff.get(file_path, {})
                    status = diff_info.get("status", "M")
                    additions = diff_info.get("additions", 0)
                    deletions = diff_info.get("deletions", 0)

                    # Create file evaluation result
                    file_evaluation = {
                        "path": file_path,
                        "status": status,
                        "additions": additions,
                        "deletions": deletions,
                        "readability": evaluation.get("readability", 5),
                        "efficiency": evaluation.get("efficiency", 5),
                        "security": evaluation.get("security", 5),
                        "structure": evaluation.get("structure", 5),
                        "error_handling": evaluation.get("error_handling", 5),
                        "documentation": evaluation.get("documentation", 5),
                        "code_style": evaluation.get("code_style", 5),
                        "overall_score": evaluation.get("overall_score", 5.0),
                        "estimated_hours": evaluation.get("estimated_hours", 0.5),
                        "summary": evaluation.get("comments", "")[:100] + "..." if len(evaluation.get("comments", "")) > 100 else evaluation.get("comments", ""),
                        "comments": evaluation.get("comments", "")
                    }

                    evaluation_results["files"].append(file_evaluation)
                    logger.info(f"Added evaluation for {file_path} with score: {file_evaluation['overall_score']}")

                # Store the whole commit evaluation for later use
                evaluation_results["whole_commit_evaluation"] = {
                    "readability": whole_commit_evaluation.get("readability", 5),
                    "efficiency": whole_commit_evaluation.get("efficiency", 5),
                    "security": whole_commit_evaluation.get("security", 5),
                    "structure": whole_commit_evaluation.get("structure", 5),
                    "error_handling": whole_commit_evaluation.get("error_handling", 5),
                    "documentation": whole_commit_evaluation.get("documentation", 5),
                    "code_style": whole_commit_evaluation.get("code_style", 5),
                    "overall_score": whole_commit_evaluation.get("overall_score", 5),
                    "effective_code_lines": whole_commit_evaluation.get("effective_code_lines", 0),
                    "non_effective_code_lines": whole_commit_evaluation.get("non_effective_code_lines", 0),
                    "estimated_hours": whole_commit_evaluation.get("estimated_hours", 0),
                    "comments": whole_commit_evaluation.get("comments", "No comments available.")
                }

                # Add the estimated working hours to the evaluation results
                evaluation_results["estimated_hours"] = whole_commit_evaluation.get("estimated_hours", 0)
                logger.info(f"Estimated working hours: {evaluation_results['estimated_hours']}")

                # Skip the additional whole commit evaluation below since we already have it
                return evaluation_results

        # Log batch evaluation settings if enabled
        elif use_batched_evaluation:
            logger.info(f"Using batched evaluation with max {max_files_per_batch} files per batch and max {max_tokens_per_batch} tokens per batch")
            print(f"Using batched evaluation (max {max_files_per_batch} files per batch, max {max_tokens_per_batch} tokens per batch)")

            # Prepare batches
            batches = []
            current_batch = []
            current_batch_tokens = 0

            # Collect file information for batching
            file_info_list = []
            for file_path, diff_info in commit_diff.items():
                # Estimate tokens
                diff_content = diff_info["diff"]
                estimated_tokens = len(diff_content.split()) * 1.2

                file_info = {
                    "file_path": file_path,
                    "file_diff": diff_content,  # Use file_diff to match _evaluate_file_batch parameter name
                    "commit": commit_hash,      # Add commit hash for reference
                    "status": diff_info["status"],
                    "additions": diff_info.get("additions", 0),
                    "deletions": diff_info.get("deletions", 0),
                    "estimated_tokens": estimated_tokens,
                    "file_size": len(diff_content)
                }
                file_info_list.append(file_info)

            # Sort by file size (smaller files first)
            file_info_list = sorted(file_info_list, key=lambda x: x["file_size"])

            # Create batches
            for file_info in file_info_list:
                # If adding this file would exceed the token limit or max files per batch, create a new batch
                if (current_batch_tokens + file_info["estimated_tokens"] > max_tokens_per_batch or
                    len(current_batch) >= max_files_per_batch) and current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_batch_tokens = 0

                current_batch.append(file_info)
                current_batch_tokens += file_info["estimated_tokens"]

            # Add the last batch if not empty
            if current_batch:
                batches.append(current_batch)

            logger.info(f"Created {len(batches)} batches for {len(file_info_list)} files")
            print(f"Created {len(batches)} batches for {len(file_info_list)} files")

            # Process each batch
            for batch_idx, batch in enumerate(batches):
                logger.info(f"Processing batch {batch_idx+1}/{len(batches)} with {len(batch)} files")
                print(f"Processing batch {batch_idx+1}/{len(batches)} with {len(batch)} files...")

                try:
                    # Use the batch evaluation method
                    batch_start_time = time.time()
                    batch_results = await self._evaluate_file_batch(batch)
                    batch_end_time = time.time()
                    logger.info(f"Batch {batch_idx+1} evaluated in {batch_end_time - batch_start_time:.2f} seconds")
                    print(f"Batch {batch_idx+1} evaluated in {batch_end_time - batch_start_time:.2f} seconds")

                    # Process batch results
                    for file_info, evaluation in zip(batch, batch_results):
                        file_path = file_info["file_path"]
                        status = file_info["status"]
                        additions = file_info["additions"]
                        deletions = file_info["deletions"]

                        # Create file evaluation result
                        file_evaluation = {
                            "path": file_path,
                            "status": status,
                            "additions": additions,
                            "deletions": deletions,
                            "readability": evaluation.get("readability", 5),
                            "efficiency": evaluation.get("efficiency", 5),
                            "security": evaluation.get("security", 5),
                            "structure": evaluation.get("structure", 5),
                            "error_handling": evaluation.get("error_handling", 5),
                            "documentation": evaluation.get("documentation", 5),
                            "code_style": evaluation.get("code_style", 5),
                            "overall_score": evaluation.get("overall_score", 5.0),
                            "estimated_hours": evaluation.get("estimated_hours", 0.5),
                            "summary": evaluation.get("comments", "")[:100] + "..." if len(evaluation.get("comments", "")) > 100 else evaluation.get("comments", ""),
                            "comments": evaluation.get("comments", "")
                        }

                        evaluation_results["files"].append(file_evaluation)
                        logger.info(f"Added evaluation for {file_path} with score: {file_evaluation['overall_score']}")

                except Exception as e:
                    logger.error(f"Error processing batch {batch_idx+1}: {str(e)}", exc_info=True)
                    print(f"Error processing batch {batch_idx+1}: {str(e)}")

                    # Create default evaluations for each file in the batch
                    for file_info in batch:
                        file_path = file_info["file_path"]
                        status = file_info["status"]
                        additions = file_info["additions"]
                        deletions = file_info["deletions"]

                        # Create default file evaluation
                        file_evaluation = {
                            "path": file_path,
                            "status": status,
                            "additions": additions,
                            "deletions": deletions,
                            "readability": 5,
                            "efficiency": 5,
                            "security": 5,
                            "structure": 5,
                            "error_handling": 5,
                            "documentation": 5,
                            "code_style": 5,
                            "overall_score": 5.0,
                            "estimated_hours": self._estimate_default_hours(additions, deletions, file_path),
                            "summary": f"Error during batch evaluation: {str(e)}",
                            "comments": f"Error during batch evaluation: {str(e)}"
                        }

                        evaluation_results["files"].append(file_evaluation)
                        logger.warning(f"Added default evaluation for {file_path} due to batch processing error")

        else:
            # Standard file-by-file evaluation
            logger.info(f"Using standard file-by-file evaluation")
            print("Using standard file-by-file evaluation")
            for i, (file_path, diff_info) in enumerate(commit_diff.items()):
                logger.info(f"Evaluating file {i+1}/{len(commit_diff)}: {file_path}")
                print(f"Evaluating file {i+1}/{len(commit_diff)}: {file_path}")
                logger.debug(f"File info: status={diff_info['status']}, additions={diff_info.get('additions', 0)}, deletions={diff_info.get('deletions', 0)}")

                # Use the new method for commit file evaluation
                start_time = time.time()
                file_evaluation = await self.evaluate_commit_file(
                    file_path,
                    diff_info["diff"],
                    diff_info["status"],
                    diff_info.get("additions", 0),
                    diff_info.get("deletions", 0),
                )
                end_time = time.time()
                logger.info(f"File {file_path} evaluated in {end_time - start_time:.2f} seconds with score: {file_evaluation.get('overall_score', 'N/A')}")
                print(f"File {file_path} evaluated in {end_time - start_time:.2f} seconds with score: {file_evaluation.get('overall_score', 'N/A')}")

                evaluation_results["files"].append(file_evaluation)
                logger.debug(f"Added evaluation for {file_path} to results")

        # Evaluate the entire commit as a whole to get estimated working hours
        logger.info("Evaluating entire commit as a whole")
        whole_commit_evaluation = await self.evaluate_commit_as_whole(commit_hash, commit_diff)

        # Add the estimated working hours to the evaluation results
        evaluation_results["estimated_hours"] = whole_commit_evaluation.get("estimated_hours", 0)
        logger.info(f"Estimated working hours: {evaluation_results['estimated_hours']}")

        # Add whole commit evaluation scores
        effective_code_lines = whole_commit_evaluation.get("effective_code_lines", int(evaluation_results['statistics']['total_additions'] * 0.6) + int(evaluation_results['statistics']['total_deletions'] * 0.6))
        non_effective_code_lines = whole_commit_evaluation.get("non_effective_code_lines", int(evaluation_results['statistics']['total_additions'] * 0.4) + int(evaluation_results['statistics']['total_deletions'] * 0.4))

        # Update statistics with effective code lines from LLM evaluation
        evaluation_results['statistics']['total_effective_lines'] = effective_code_lines

        evaluation_results["whole_commit_evaluation"] = {
            "readability": whole_commit_evaluation.get("readability", 5),
            "efficiency": whole_commit_evaluation.get("efficiency", 5),
            "security": whole_commit_evaluation.get("security", 5),
            "structure": whole_commit_evaluation.get("structure", 5),
            "error_handling": whole_commit_evaluation.get("error_handling", 5),
            "documentation": whole_commit_evaluation.get("documentation", 5),
            "code_style": whole_commit_evaluation.get("code_style", 5),
            "overall_score": whole_commit_evaluation.get("overall_score", 5),
            "effective_code_lines": effective_code_lines,
            "non_effective_code_lines": non_effective_code_lines,
            "comments": whole_commit_evaluation.get("comments", "No comments available.")
        }

        # Generate overall summary
        logger.info(f"Generating overall summary for commit {commit_hash}")
        summary_prompt = self._create_summary_prompt(evaluation_results)
        logger.debug(f"Summary prompt size: {len(summary_prompt)} characters")

        # Use agenerate instead of ainvoke
        messages = [HumanMessage(content=summary_prompt)]
        logger.info("Sending summary request to model")
        start_time = time.time()
        # Get user message for logging
        user_message = messages[0].content

        # Call the model
        summary_response = await self.model.agenerate(messages=[messages])
        end_time = time.time()
        logger.info(f"Summary response received in {end_time - start_time:.2f} seconds")

        summary_text = summary_response.generations[0][0].text

        # Log both prompt and response to the same file
        log_llm_interaction(user_message, summary_text, interaction_type="summary",
                          extra_info=f"Commit: {commit_hash}, Files: {len(evaluation_results['files'])}")
        logger.debug(f"Summary text size: {len(summary_text)} characters")
        logger.debug(f"Summary text (first 100 chars): {summary_text[:100]}...")

        evaluation_results["summary"] = summary_text
        logger.info(f"Evaluation for commit {commit_hash} completed successfully")

        return evaluation_results

    def _create_summary_prompt(self, evaluation_results: Dict[str, Any]) -> str:
        """Create a prompt for generating the overall commit summary."""
        files_summary = "\n".join(
            f"- {file['path']} ({file['status']}): {file['summary']}"
            for file in evaluation_results["files"]
        )

        # Include whole commit evaluation if available
        whole_commit_evaluation = ""
        if "whole_commit_evaluation" in evaluation_results:
            eval_data = evaluation_results["whole_commit_evaluation"]

            # Include effective and non-effective code lines if available
            code_lines_info = ""
            if "effective_code_lines" in eval_data or "non_effective_code_lines" in eval_data:
                effective_lines = eval_data.get('effective_code_lines', 'N/A')
                non_effective_lines = eval_data.get('non_effective_code_lines', 'N/A')
                total_lines = (effective_lines if isinstance(effective_lines, int) else 0) + (non_effective_lines if isinstance(non_effective_lines, int) else 0)

                if total_lines > 0 and isinstance(effective_lines, int) and isinstance(non_effective_lines, int):
                    effective_percentage = (effective_lines / total_lines) * 100 if total_lines > 0 else 0
                    code_lines_info = f"""
- Effective Code Lines: {effective_lines} ({effective_percentage:.1f}% of total changes)
- Non-Effective Code Lines: {non_effective_lines} ({100 - effective_percentage:.1f}% of total changes)"""
                else:
                    code_lines_info = f"""
- Effective Code Lines: {effective_lines}
- Non-Effective Code Lines: {non_effective_lines}"""

            whole_commit_evaluation = f"""
Whole Commit Evaluation:
- Readability: {eval_data.get('readability', 'N/A')}/10
- Efficiency: {eval_data.get('efficiency', 'N/A')}/10
- Security: {eval_data.get('security', 'N/A')}/10
- Structure: {eval_data.get('structure', 'N/A')}/10
- Error Handling: {eval_data.get('error_handling', 'N/A')}/10
- Documentation: {eval_data.get('documentation', 'N/A')}/10
- Code Style: {eval_data.get('code_style', 'N/A')}/10
- Overall Score: {eval_data.get('overall_score', 'N/A')}/10{code_lines_info}
- Comments: {eval_data.get('comments', 'No comments available.')}
"""

        # Include estimated working hours if available
        estimated_hours = ""
        if "estimated_hours" in evaluation_results:
            estimated_hours = f"- Estimated working hours (for 5-10+ years experienced developer): {evaluation_results['estimated_hours']} hours\n"

        return f"""Please provide a concise summary of this commit's changes:

Files modified:
{files_summary}

Statistics:
- Total files: {evaluation_results['statistics']['total_files']}
- Total additions: {evaluation_results['statistics']['total_additions']}
- Total deletions: {evaluation_results['statistics']['total_deletions']}
{estimated_hours}
{whole_commit_evaluation}
Please provide a brief summary of the overall changes and their impact.
If estimated working hours are provided, please comment on whether this estimate seems reasonable given the scope of changes."""


def generate_evaluation_markdown(evaluation_results: List[FileEvaluationResult]) -> str:
    """
    生成评价结果的Markdown表格，按提交组织而不是按文件组织

    Args:
        evaluation_results: 文件评价结果列表

    Returns:
        str: Markdown格式的评价表格
    """
    if not evaluation_results:
        return "## 代码评价结果\n\n没有找到需要评价的代码提交。"

    # 按日期排序结果
    sorted_results = sorted(evaluation_results, key=lambda x: x.date)

    # Create Markdown header
    markdown = "# Code Evaluation Report\n\n"

    # Add overview
    author = sorted_results[0].author if sorted_results else "Unknown"
    start_date = sorted_results[0].date.strftime("%Y-%m-%d") if sorted_results else "Unknown"
    end_date = sorted_results[-1].date.strftime("%Y-%m-%d") if sorted_results else "Unknown"

    # 按提交组织结果
    commits_dict = {}
    for result in sorted_results:
        commit_hash = result.commit_hash
        if commit_hash not in commits_dict:
            commits_dict[commit_hash] = {
                "hash": commit_hash,
                "message": result.commit_message,
                "date": result.date,
                "author": result.author,
                "files": [],
                "scores": {
                    "readability": 0,
                    "efficiency": 0,
                    "security": 0,
                    "structure": 0,
                    "error_handling": 0,
                    "documentation": 0,
                    "code_style": 0,
                    "overall_score": 0,
                },
                "estimated_hours": 0,  # 每个提交只计算一次工作时间
                "has_estimated_hours": False,  # 标记是否有工作时间估算
            }

        # 添加文件到提交
        commits_dict[commit_hash]["files"].append({
            "file_path": result.file_path,
            "evaluation": result.evaluation
        })

        # 累加分数
        eval = result.evaluation
        commits_dict[commit_hash]["scores"]["readability"] += eval.readability
        commits_dict[commit_hash]["scores"]["efficiency"] += eval.efficiency
        commits_dict[commit_hash]["scores"]["security"] += eval.security
        commits_dict[commit_hash]["scores"]["structure"] += eval.structure
        commits_dict[commit_hash]["scores"]["error_handling"] += eval.error_handling
        commits_dict[commit_hash]["scores"]["documentation"] += eval.documentation
        commits_dict[commit_hash]["scores"]["code_style"] += eval.code_style
        commits_dict[commit_hash]["scores"]["overall_score"] += eval.overall_score

        # 只计算一次工作时间（使用第一个文件的工作时间估算）
        if hasattr(eval, 'estimated_hours') and eval.estimated_hours and not commits_dict[commit_hash]["has_estimated_hours"]:
            commits_dict[commit_hash]["estimated_hours"] = eval.estimated_hours
            commits_dict[commit_hash]["has_estimated_hours"] = True

    # 按日期排序提交
    sorted_commits = sorted(commits_dict.values(), key=lambda x: x["date"])

    # 计算总体统计
    total_files = len(sorted_results)
    total_commits = len(sorted_commits)
    total_scores = {
        "readability": 0,
        "efficiency": 0,
        "security": 0,
        "structure": 0,
        "error_handling": 0,
        "documentation": 0,
        "code_style": 0,
        "overall_score": 0,
    }
    total_estimated_hours = 0

    for commit in sorted_commits:
        file_count = len(commit["files"])
        # 计算每个提交的平均分
        for key in total_scores.keys():
            commit["scores"][key] = commit["scores"][key] / file_count
            total_scores[key] += commit["scores"][key]

        # 累加工作时间
        total_estimated_hours += commit["estimated_hours"]

    # 计算平均分
    avg_scores = {k: v / total_commits for k, v in total_scores.items()} if total_commits > 0 else {k: 0 for k in total_scores.keys()}

    # 添加概览
    markdown += f"## Overview\n\n"
    markdown += f"- **Developer**: {author}\n"
    markdown += f"- **Time Range**: {start_date} to {end_date}\n"
    markdown += f"- **Commits Evaluated**: {total_commits}\n"
    markdown += f"- **Files Evaluated**: {total_files}\n"

    # 添加总工作时间
    if total_estimated_hours > 0:
        markdown += f"- **Total Estimated Working Hours**: {total_estimated_hours:.1f} hours\n"

    markdown += "\n"

    # 添加平均分数
    markdown += "## Overall Scores\n\n"
    markdown += "| Dimension | Average Score |\n"
    markdown += "|-----------|---------------|\n"
    markdown += f"| Readability | {avg_scores['readability']:.1f} |\n"
    markdown += f"| Efficiency & Performance | {avg_scores['efficiency']:.1f} |\n"
    markdown += f"| Security | {avg_scores['security']:.1f} |\n"
    markdown += f"| Structure & Design | {avg_scores['structure']:.1f} |\n"
    markdown += f"| Error Handling | {avg_scores['error_handling']:.1f} |\n"
    markdown += f"| Documentation & Comments | {avg_scores['documentation']:.1f} |\n"
    markdown += f"| Code Style | {avg_scores['code_style']:.1f} |\n"
    markdown += f"| **Overall Score** | **{avg_scores['overall_score']:.1f}** |\n"

    # 添加平均工作时间
    if total_estimated_hours > 0:
        markdown += f"| **Avg. Estimated Hours/Commit** | **{total_estimated_hours / total_commits:.1f}** |\n"

    markdown += "\n"

    # 添加质量评估
    overall_score = avg_scores["overall_score"]
    quality_level = ""
    if overall_score >= 9.0:
        quality_level = "Exceptional"
    elif overall_score >= 7.0:
        quality_level = "Excellent"
    elif overall_score >= 5.0:
        quality_level = "Good"
    elif overall_score >= 3.0:
        quality_level = "Needs Improvement"
    else:
        quality_level = "Poor"

    markdown += f"**Overall Code Quality**: {quality_level}\n\n"

    # 添加提交评价详情
    markdown += "## 提交评价详情\n\n"

    for idx, commit in enumerate(sorted_commits, 1):
        # 提交标题
        short_hash = commit["hash"][:8]
        markdown += f"### {idx}. Commit {short_hash}: {commit['message'].split('\n')[0]}\n\n"

        # 提交基本信息
        markdown += f"- **Full Hash**: {commit['hash']}\n"
        markdown += f"- **Date**: {commit['date'].strftime('%Y-%m-%d %H:%M')}\n"
        markdown += f"- **Author**: {commit['author']}\n"
        markdown += f"- **Files Modified**: {len(commit['files'])}\n"

        # 添加工作时间估算
        if commit["has_estimated_hours"]:
            markdown += f"- **Estimated Working Hours**: {commit['estimated_hours']:.1f}\n"

        # 添加提交平均分数
        markdown += "\n**Commit Scores**:\n\n"
        markdown += "| Dimension | Score |\n"
        markdown += "|----------|------|\n"
        markdown += f"| Readability | {commit['scores']['readability']:.1f} |\n"
        markdown += f"| Efficiency & Performance | {commit['scores']['efficiency']:.1f} |\n"
        markdown += f"| Security | {commit['scores']['security']:.1f} |\n"
        markdown += f"| Structure & Design | {commit['scores']['structure']:.1f} |\n"
        markdown += f"| Error Handling | {commit['scores']['error_handling']:.1f} |\n"
        markdown += f"| Documentation & Comments | {commit['scores']['documentation']:.1f} |\n"
        markdown += f"| Code Style | {commit['scores']['code_style']:.1f} |\n"
        markdown += f"| **Overall Score** | **{commit['scores']['overall_score']:.1f}** |\n\n"

        # 添加修改的文件列表
        markdown += "**Modified Files**:\n\n"
        for file_idx, file in enumerate(commit["files"], 1):
            file_path = file["file_path"]
            eval = file["evaluation"]

            markdown += f"{file_idx}. **{file_path}**\n"
            markdown += f"   - Score: {eval.overall_score:.1f}\n"

            # 添加文件评论摘要（只显示前100个字符）
            comments_summary = eval.comments[:100] + "..." if len(eval.comments) > 100 else eval.comments
            comments_summary = comments_summary.replace("\n", " ").strip()
            markdown += f"   - Comments: {comments_summary}\n\n"

        # 添加详细评论
        markdown += "**Detailed Comments**:\n\n"
        for file_idx, file in enumerate(commit["files"], 1):
            file_path = file["file_path"]
            eval = file["evaluation"]

            markdown += f"**File {file_idx}: {file_path}**\n\n"

            # 检查评论是否包含调整说明
            if "**Note**: Scores have been adjusted for differentiation" in eval.comments:
                # 分离评论和调整说明
                comments_parts = eval.comments.split("**Note**: Scores have been adjusted for differentiation")
                main_comments = comments_parts[0].strip()
                adjustment_note = "**Note**: Scores have been adjusted for differentiation" + comments_parts[1]

                # 添加主要评论
                markdown += f"{main_comments}\n\n"

                # 添加调整说明（使用特殊格式）
                markdown += f"<div style='background-color: #f8f9fa; padding: 10px; border-left: 4px solid #007bff; margin-bottom: 10px;'>{adjustment_note}</div>\n\n"
            else:
                markdown += f"{eval.comments}\n\n"

        markdown += "---\n\n"

    return markdown