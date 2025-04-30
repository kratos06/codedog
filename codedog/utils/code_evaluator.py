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

def process(generated_text: str) -> dict:
    """
    Process the JSON response from LLM evaluation.

    Args:
        generated_text: The raw text response from the LLM

    Returns:
        Dictionary containing the processed evaluation data

    Raises:
        Exception: If the response cannot be parsed as valid JSON
    """
    logger.info(f"Processing LLM response text of length: {len(generated_text)}")

    # First try: direct JSON parsing of the entire text
    try:
        data = json.loads(generated_text)
        logger.info("Successfully parsed entire text as JSON")
        return data
    except json.JSONDecodeError:
        logger.info("Failed to parse entire text as JSON, trying to extract JSON content")

    # Second try: extract JSON from code blocks
    try:
        # Look for JSON in code blocks (```json ... ```)
        code_block_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
        code_blocks = re.findall(code_block_pattern, generated_text)

        if code_blocks:
            logger.info(f"Found {len(code_blocks)} code blocks, trying to parse each")

            for block in code_blocks:
                try:
                    data = json.loads(block)
                    logger.info("Successfully parsed JSON from code block")
                    return data
                except json.JSONDecodeError:
                    # 尝试清理后再解析
                    cleaned_block = _clean_json_string(block)
                    try:
                        data = json.loads(cleaned_block)
                        logger.info("Successfully parsed JSON from cleaned code block")
                        return data
                    except json.JSONDecodeError:
                        continue  # Try next block
    except Exception as e:
        logger.warning(f"Error while extracting code blocks: {e}")

    # Third try: more precise JSON pattern matching
    try:
        # More precise pattern to find JSON objects
        # This looks for patterns that start with { and end with } with balanced braces
        json_pattern = r'(\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\})'
        json_matches = re.findall(json_pattern, generated_text)

        if json_matches:
            logger.info(f"Found {len(json_matches)} potential JSON objects, trying to parse each")

            # 按长度排序，尝试最长的匹配（可能是完整的JSON）
            for json_str in sorted(json_matches, key=len, reverse=True):
                try:
                    # Try to parse each match
                    data = json.loads(json_str)
                    logger.info("Successfully parsed JSON from pattern match")
                    return data
                except json.JSONDecodeError:
                    # 尝试清理后再解析
                    cleaned_json = _clean_json_string(json_str)
                    try:
                        data = json.loads(cleaned_json)
                        logger.info("Successfully parsed JSON after cleaning pattern match")
                        return data
                    except json.JSONDecodeError:
                        continue  # Try next match
    except Exception as e:
        logger.warning(f"Error while matching JSON pattern: {e}")

    # Fourth try: clean the text and try again with simpler pattern
    try:
        # Remove markdown and code block markers
        cleaned_text = generated_text.replace("```json", "").replace("```", "")

        # Try a simpler pattern that might catch more JSON-like structures
        simple_json_pattern = r'(\{[\s\S]*?\})'
        json_matches = re.findall(simple_json_pattern, cleaned_text)

        if json_matches:
            logger.info(f"Found {len(json_matches)} potential JSON objects in cleaned text")

            for json_str in sorted(json_matches, key=len, reverse=True):  # Try longest matches first
                # Apply extensive cleaning
                cleaned_json = _clean_json_string(json_str)

                try:
                    data = json.loads(cleaned_json)
                    logger.info("Successfully parsed JSON after extensive cleaning")
                    return data
                except json.JSONDecodeError:
                    continue  # Try next match
    except Exception as e:
        logger.warning(f"Error while cleaning and parsing JSON: {e}")

    # 最后尝试：直接解析整个文本，但先进行清理
    try:
        cleaned_text = _clean_json_string(generated_text)
        data = json.loads(cleaned_text)
        logger.info("Successfully parsed entire cleaned text as JSON")
        return data
    except json.JSONDecodeError:
        pass

    # 如果所有尝试都失败，返回一个空的默认结果而不是抛出异常
    logger.error("Failed to extract valid JSON, returning default empty result")
    return {
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
        "effective_additions": 0,
        "effective_deletions": 0,
        "estimated_hours": 0.5,
        "comments": "Failed to parse JSON response from LLM. The code may require manual review.",
        "file_evaluations": {}
    }


def _clean_json_string(json_str: str) -> str:
    """
    Apply extensive cleaning to a JSON string to make it parseable.

    Args:
        json_str: The JSON string to clean

    Returns:
        A cleaned JSON string that might be parseable
    """
    try:
        # 移除控制字符
        cleaned = ''.join(ch for ch in json_str if ch >= ' ' or ch in '\t\n\r')

        # 基本清理
        cleaned = cleaned.strip()

        # 处理单引号
        # 1. 先替换属性名中的单引号
        cleaned = re.sub(r'([{,]\s*)\'([^\']+?)\'(\s*:)', r'\1"\2"\3', cleaned)

        # 2. 再替换字符串值中的单引号
        cleaned = re.sub(r':\s*\'([^\']*?)\'([,}])', r':"\1"\2', cleaned)

        # 移除尾部逗号
        cleaned = re.sub(r',\s*}', '}', cleaned)
        cleaned = re.sub(r',\s*]', ']', cleaned)

        # 修复缺少引号的属性名
        cleaned = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', cleaned)

        # 处理布尔值和null
        # 1. 先将未加引号的布尔值和null加上引号
        cleaned = re.sub(r':\s*(true|false|null)([,}])', r':"\1"\2', cleaned)

        # 2. 再将加了引号的布尔值和null恢复为正确的JSON格式
        cleaned = re.sub(r':"true"', r':true', cleaned)
        cleaned = re.sub(r':"false"', r':false', cleaned)
        cleaned = re.sub(r':"null"', r':null', cleaned)

        # 处理数字
        # 确保数字值不被引号包围
        def replace_quoted_numbers(match):
            num = match.group(2)
            try:
                float(num)  # 检查是否为有效数字
                return f'{match.group(1)}{num}{match.group(3)}'
            except ValueError:
                return match.group(0)

        cleaned = re.sub(r'(:\s*)"(\d+|\d+\.\d+)"([,}])', replace_quoted_numbers, cleaned)

        # 处理转义字符
        # 我们不需要添加额外的转义，因为这可能会导致问题
        # 只需确保现有的转义是正确的
        cleaned = cleaned.replace('\\\\"', '\\"')  # 修复重复转义

        # 记录清理结果
        logger.debug(f"Original JSON string length: {len(json_str)}")
        logger.debug(f"Cleaned JSON string length: {len(cleaned)}")

        return cleaned
    except Exception as e:
        logger.error(f"Error cleaning JSON string: {e}")
        return json_str  # 如果清理失败，返回原始字符串

# 跟踪日志文件是否已初始化（每次程序运行时重置）
_llm_log_initialized = False

# Function to log LLM inputs and outputs to a single file
def log_llm_interaction(prompt, response, interaction_type="default", extra_info=None):
    """
    Log LLM prompts and responses to a single LLM_interactions.log file
    Each time the program runs, the log file is recreated (not appended to)

    Args:
        prompt: The prompt sent to the LLM
        response: The response received from the LLM
        interaction_type: A label to identify the type of interaction (e.g., "file_evaluation", "summary")
        extra_info: Optional additional information to log (e.g., commit details for whole-commit evaluation)
    """
    global _llm_log_initialized
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # 确定文件打开模式：第一次调用时使用"w"创建新文件，后续调用使用"a"追加
    file_mode = "a" if _llm_log_initialized else "w"

    # 标记日志文件已初始化
    if not _llm_log_initialized:
        _llm_log_initialized = True
        logger.info("Initialized new LLM interaction log file")

    # Log both prompt and response to the same file
    with open("logs/LLM_interactions.log", file_mode, encoding="utf-8") as f:
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


@dataclass(frozen=False)  # Allow modification for commit_evaluation
class FileEvaluationResult:
    """文件评价结果"""
    file_path: str
    commit_hash: str
    commit_message: str
    date: datetime
    author: str
    evaluation: CodeEvaluation = None
    commit_evaluation: Dict[str, Any] = None  # For whole-commit evaluation results


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
  "effective_additions": number,
  "effective_deletions": number,
  "estimated_hours": number,
  "comments": "detailed analysis with specific observations and recommendations",
  "file_evaluations": {
    "file_path_1": {
      "readability": score,
      "efficiency": score,
      "security": score,
      "structure": score,
      "error_handling": score,
      "documentation": score,
      "code_style": score,
      "overall_score": score,
      "estimated_hours": number,
      "comments": "brief analysis of this specific file"
    },
    "file_path_2": {
      // same structure as above
    }
    // additional files...
  }
}
```

## JSON Output Guidelines:
1. All scores MUST be integers or decimals between 1-10
2. The overall_score should reflect the weighted importance of all dimensions
3. effective_code_lines should count ONLY changes that affect behavior or functionality
4. non_effective_code_lines should count formatting, style, and cosmetic changes
5. effective_additions should count ONLY added lines that contain actual logic/functionality
6. effective_deletions should count ONLY deleted lines that contained actual logic/functionality
7. estimated_hours should be a realistic estimate for an experienced programmer (5-10+ years)
8. comments should include:
   - Specific observations about code quality
   - Concrete recommendations for improvement
   - Explanation of effective vs. non-effective changes
   - Justification for your working hours estimate
   - Any security concerns or performance issues
   - Suggestions for better practices or patterns
9. file_evaluations should include evaluations for each significant file changed

IMPORTANT: Ensure your response is valid JSON that can be parsed programmatically. Do not include explanatory text outside the JSON structure.
CRITICAL: The fields effective_additions, effective_deletions, and file_evaluations are REQUIRED and must be included in your response.
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
                            elif "overall" in comments_dict and isinstance(comments_dict["overall"], str):
                                # 如果overall是字符串，直接使用
                                comments_str = comments_dict["overall"]
                            else:
                                # 否则，尝试从各个评分字段中提取评论
                                for field in ["readability", "efficiency", "security", "structure", "error_handling", "documentation", "code_style"]:
                                    if field in comments_dict:
                                        if isinstance(comments_dict[field], dict) and "comment" in comments_dict[field]:
                                            comments_str += f"{field.capitalize()}: {comments_dict[field]['comment']}\n"
                                        elif isinstance(comments_dict[field], str) and len(comments_dict[field]) > 5:
                                            # 只添加看起来像评论的字符串（长度>5）
                                            comments_str += f"{field.capitalize()}: {comments_dict[field]}\n"

                            # 检查是否有summary或description字段
                            for field in ["summary", "description", "analysis", "evaluation"]:
                                if field in comments_dict and isinstance(comments_dict[field], str) and len(comments_dict[field]) > 10:
                                    if comments_str:
                                        comments_str += f"\n\n{field.capitalize()}:\n{comments_dict[field]}"
                                    else:
                                        comments_str = comments_dict[field]

                            # 如果没有找到任何评论，尝试直接将字典转换为字符串
                            if not comments_str:
                                try:
                                    comments_str = json.dumps(comments_dict, ensure_ascii=False, indent=2)
                                except Exception:
                                    # 如果JSON序列化失败，使用str()
                                    comments_str = str(comments_dict)

                            normalized_result["comments"] = comments_str
                        elif isinstance(normalized_result["comments"], list):
                            # 如果是列表，尝试连接所有项
                            comments_list = []
                            for item in normalized_result["comments"]:
                                if isinstance(item, str):
                                    comments_list.append(item)
                                elif isinstance(item, dict) and "comment" in item:
                                    comments_list.append(item["comment"])
                                elif isinstance(item, dict):
                                    # 尝试找到看起来像评论的字段
                                    for key, value in item.items():
                                        if isinstance(value, str) and len(value) > 10:
                                            comments_list.append(f"{key}: {value}")
                                            break
                                    else:
                                        # 如果没有找到合适的字段，使用整个字典
                                        comments_list.append(str(item))
                                else:
                                    comments_list.append(str(item))

                            normalized_result["comments"] = "\n\n".join(comments_list)
                        else:
                            # 如果不是字典或列表，尝试转换为字符串
                            normalized_result["comments"] = str(normalized_result["comments"])
                    except Exception as e:
                        logger.error(f"Error converting comments to string: {e}")
                        logger.error(f"Original comments value: {normalized_result['comments']}")
                        normalized_result["comments"] = f"Error converting evaluation comments: {str(e)}. The code may require manual review."

                # 确保comments不包含可能导致JSON解析问题的控制字符
                if isinstance(normalized_result["comments"], str):
                    normalized_result["comments"] = ''.join(ch for ch in normalized_result["comments"]
                                                          if ch >= ' ' or ch in '\t\n\r')

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
            "effective_additions": 0,  # Add effective_additions field
            "effective_deletions": 0,  # Add effective_deletions field
            "estimated_hours": 0.0,
            "comments": f"Evaluation failed: {error_message}. The code may require manual review.",
            "file_evaluations": {}  # Add empty file_evaluations field
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



    def _sanitize_content(self, content: str) -> str:
        """清理内容中的异常字符，确保内容可以安全地发送到API。

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
            for i, char in enumerate(content):
                # 检查字符的Unicode码点
                code_point = ord(char)

                # 保留基本可打印字符和常用空白字符
                if (char.isprintable() or char in [' ', '\n', '\t', '\r']) and code_point < 65536:
                    sanitized += char
                else:
                    # 记录并替换不可打印字符或高Unicode码点字符
                    if i < 1000 or i % 1000 == 0:  # 限制日志输出
                        print(f"DEBUG: Replaced invalid character at position {i}, Unicode: U+{code_point:04X}")
                    sanitized += ' '

            # 额外清理可能导致JSON解析问题的字符
            sanitized = sanitized.replace('\u2028', ' ')  # 行分隔符
            sanitized = sanitized.replace('\u2029', ' ')  # 段落分隔符
            sanitized = sanitized.replace('\uFEFF', '')   # 零宽不换行空格

            # 清理其他可能导致问题的不可见Unicode字符
            for c in ['\u200B', '\u200C', '\u200D', '\u200E', '\u200F']:
                sanitized = sanitized.replace(c, '')

            # 如果清理后的内容太短，返回一个提示
            if len(sanitized.strip()) < 10:
                return "代码内容太短或为空，无法进行有效评估。"

            return sanitized
        except Exception as e:
            print(f"DEBUG: Error sanitizing content: {e}")
            logger.error(f"Error sanitizing content: {e}")
            # 如果清理过程出错，返回一个安全的默认字符串
            return "内容清理过程中出错，无法处理。"

    def _clean_control_chars(self, json_str: str) -> str:
        """清理JSON字符串中的无效控制字符。

        Args:
            json_str: 包含无效控制字符的JSON字符串

        Returns:
            str: 清理后的JSON字符串
        """
        try:
            # 打印原始字符串的十六进制表示，帮助调试
            print(f"DEBUG: Cleaning control characters in JSON string")
            logger.debug(f"Detected invalid control characters, attempting to clean")

            # 检测并记录异常字符
            has_invalid_chars = False
            for i, char in enumerate(json_str[:1000]):  # 只检查前1000个字符以避免日志过大
                if ord(char) < 32 and char not in ['\n', '\r', '\t']:
                    has_invalid_chars = True
                    print(f"DEBUG: Found invalid control character at position {i}, ASCII: {ord(char)}")
                    logger.warning(f"Invalid control character at position {i}, ASCII: {ord(char)}")
                elif ord(char) > 127 and ord(char) < 160:  # Latin-1 控制字符
                    has_invalid_chars = True
                    print(f"DEBUG: Found Latin-1 control character at position {i}, Unicode: U+{ord(char):04X}")
                    logger.warning(f"Latin-1 control character at position {i}, Unicode: U+{ord(char):04X}")

            if has_invalid_chars:
                print(f"DEBUG: Invalid control characters detected in JSON string")

            # 移除或替换所有控制字符（ASCII 0-31），但保留必要的空白字符
            cleaned = ""
            for i, char in enumerate(json_str):
                code_point = ord(char)

                # 处理各种类型的控制字符
                if code_point < 32:  # ASCII控制字符
                    if char in ['\n', '\r', '\t']:  # 保留这些空白字符
                        cleaned += char
                    else:
                        # 跳过其他控制字符，并记录位置
                        if i < 1000 or i % 1000 == 0:  # 限制日志输出
                            print(f"DEBUG: Removed control character at position {i}, ASCII: {code_point}")
                elif code_point > 127 and code_point < 160:  # Latin-1 控制字符
                    # 跳过Latin-1控制字符
                    if i < 1000 or i % 1000 == 0:  # 限制日志输出
                        print(f"DEBUG: Removed Latin-1 control character at position {i}, Unicode: U+{code_point:04X}")
                elif code_point > 8000:  # 高Unicode码点字符，可能导致问题
                    # 替换高Unicode码点字符
                    if i < 1000 or i % 1000 == 0:  # 限制日志输出
                        print(f"DEBUG: Replaced high Unicode character at position {i}, Unicode: U+{code_point:04X}")
                    cleaned += ' '
                else:
                    cleaned += char

            # 特别处理转义序列
            # 确保所有反斜杠后面的字符都是有效的转义字符
            cleaned = re.sub(r'\\([^"\\/bfnrtu])', r'\\\\\1', cleaned)

            # 处理JSON中的特殊字符
            cleaned = cleaned.replace('\b', '\\b')
            cleaned = cleaned.replace('\f', '\\f')

            # 额外清理可能导致JSON解析问题的字符
            cleaned = cleaned.replace('\u2028', ' ')  # 行分隔符
            cleaned = cleaned.replace('\u2029', ' ')  # 段落分隔符
            cleaned = cleaned.replace('\uFEFF', '')   # 零宽不换行空格

            # 清理其他可能导致问题的不可见Unicode字符
            for c in ['\u200B', '\u200C', '\u200D', '\u200E', '\u200F']:
                cleaned = cleaned.replace(c, '')

            # 修复常见的JSON格式问题
            # 1. 确保属性名称有双引号
            cleaned = re.sub(r'([{,])\s*(\w+)\s*:', r'\1"\2":', cleaned)

            # 2. 修复缺失的逗号
            cleaned = re.sub(r'(true|false|null|"[^"]*"|[0-9]+)\s*([{[])', r'\1,\2', cleaned)
            cleaned = re.sub(r'(true|false|null|"[^"]*"|[0-9]+)\s*("[\w]+"\s*:)', r'\1,\2', cleaned)

            # 3. 修复多余的逗号
            cleaned = re.sub(r',\s*([}\]])', r'\1', cleaned)

            # 4. 修复缺失的引号
            cleaned = re.sub(r':\s*([^"{}\[\],\s][^{}\[\],\s]*)\s*([,}])', r':"\1"\2', cleaned)

            # 尝试验证JSON是否有效
            try:
                json.loads(cleaned)
                print(f"DEBUG: Successfully cleaned and validated JSON string")
                logger.info("Successfully cleaned and validated JSON string")
            except json.JSONDecodeError as json_error:
                print(f"DEBUG: Cleaned JSON is still invalid: {json_error}")
                logger.warning(f"Cleaned JSON is still invalid: {json_error}")

            return cleaned
        except Exception as e:
            logger.error(f"Error cleaning control characters: {e}")
            print(f"DEBUG: Error cleaning control characters: {e}")
            # 如果清理过程出错，返回原始字符串
            return json_str

    def _process_llm_response(self, response_text: str) -> Dict[str, Any]:
        """
        Process LLM response text into evaluation data format.

        Args:
            response_text: The raw response text from LLM

        Returns:
            Dict containing processed evaluation data
        """
        try:
            # 尝试解析JSON响应
            try:
                # 首先尝试直接解析
                data = json.loads(response_text)
                logger.info("Successfully parsed LLM response as JSON")
            except json.JSONDecodeError:
                # 如果直接解析失败，尝试使用process函数提取和清理JSON
                logger.warning("Failed to parse LLM response directly, trying to extract JSON")
                data = process(response_text)
                logger.info("Successfully extracted and parsed JSON from LLM response")

            # 记录原始数据结构
            logger.debug(f"Parsed data structure: {list(data.keys())}")

            # 提取主要评估指标，确保类型转换安全
            eval_data = {}

            # 处理分数字段
            score_fields = ["readability", "efficiency", "security", "structure",
                           "error_handling", "documentation", "code_style"]

            for field in score_fields:
                try:
                    value = data.get(field, 5)
                    # 确保值是数字
                    if isinstance(value, str):
                        value = float(value.strip())
                    eval_data[field] = int(round(float(value)))
                except (ValueError, TypeError):
                    logger.warning(f"Invalid value for {field}: {data.get(field)}, using default 5")
                    eval_data[field] = 5

            # 处理overall_score
            try:
                overall = data.get("overall_score", 5.0)
                if isinstance(overall, str):
                    overall = float(overall.strip())
                eval_data["overall_score"] = float(round(float(overall), 1))
            except (ValueError, TypeError):
                logger.warning(f"Invalid value for overall_score: {data.get('overall_score')}, using default 5.0")
                eval_data["overall_score"] = 5.0

            # 处理数值字段
            numeric_fields = [
                ("effective_code_lines", 0, int),
                ("non_effective_code_lines", 0, int),
                ("effective_additions", 0, int),
                ("effective_deletions", 0, int),
                ("estimated_hours", 0.5, float)
            ]

            for field_name, default_value, convert_func in numeric_fields:
                try:
                    value = data.get(field_name, default_value)
                    if isinstance(value, str):
                        value = float(value.strip())
                    eval_data[field_name] = convert_func(value)
                except (ValueError, TypeError):
                    logger.warning(f"Invalid value for {field_name}: {data.get(field_name)}, using default {default_value}")
                    eval_data[field_name] = default_value

            # 处理comments字段
            eval_data["comments"] = data.get("comments", "No evaluation comments provided.")

            # 处理file_evaluations字段（如果存在）
            if "file_evaluations" in data:
                file_evals = {}
                for file_path, file_data in data["file_evaluations"].items():
                    file_eval = {}

                    # 处理文件评估的分数字段
                    for field in score_fields:
                        try:
                            value = file_data.get(field, 5)
                            if isinstance(value, str):
                                value = float(value.strip())
                            file_eval[field] = int(round(float(value)))
                        except (ValueError, TypeError):
                            file_eval[field] = 5

                    # 处理文件评估的overall_score
                    try:
                        overall = file_data.get("overall_score", 5.0)
                        if isinstance(overall, str):
                            overall = float(overall.strip())
                        file_eval["overall_score"] = float(round(float(overall), 1))
                    except (ValueError, TypeError):
                        file_eval["overall_score"] = 5.0

                    # 处理文件评估的其他字段
                    try:
                        effective_lines = file_data.get("effective_lines", 0)
                        if isinstance(effective_lines, str):
                            effective_lines = int(effective_lines.strip())
                        file_eval["effective_lines"] = int(effective_lines)
                    except (ValueError, TypeError):
                        file_eval["effective_lines"] = 0

                    # 添加comments字段（如果存在）
                    if "comments" in file_data:
                        file_eval["comments"] = file_data["comments"]

                    file_evals[file_path] = file_eval

                eval_data["file_evaluations"] = file_evals

            # 验证分数在有效范围内
            for field in score_fields:
                if eval_data[field] < 1:
                    eval_data[field] = 1
                elif eval_data[field] > 10:
                    eval_data[field] = 10

            if eval_data["overall_score"] < 1.0:
                eval_data["overall_score"] = 1.0
            elif eval_data["overall_score"] > 10.0:
                eval_data["overall_score"] = 10.0

            # 确保estimated_hours为正值
            if eval_data["estimated_hours"] < 0:
                eval_data["estimated_hours"] = 0.5

            # 确保comments字段是字符串类型
            if not isinstance(eval_data["comments"], str):
                try:
                    if isinstance(eval_data["comments"], dict):
                        # 处理嵌套的comments字典
                        comments_dict = eval_data["comments"]
                        comments_str = ""

                        # 尝试提取评论内容
                        for field in ["overall", "summary", "description", "analysis"]:
                            if field in comments_dict and isinstance(comments_dict[field], str) and len(comments_dict[field]) > 10:
                                comments_str = comments_dict[field]
                                break

                        # 如果没有找到主要评论，尝试从各个字段提取
                        if not comments_str:
                            for field, value in comments_dict.items():
                                if isinstance(value, str) and len(value) > 10:
                                    comments_str += f"{field.capitalize()}: {value}\n\n"

                        # 如果仍然没有找到评论，将整个字典转换为字符串
                        if not comments_str:
                            comments_str = json.dumps(comments_dict, ensure_ascii=False, indent=2)

                        eval_data["comments"] = comments_str
                    elif isinstance(eval_data["comments"], list):
                        # 如果是列表，尝试连接所有项
                        comments_list = []
                        for item in eval_data["comments"]:
                            if isinstance(item, str):
                                comments_list.append(item)
                            elif isinstance(item, dict):
                                # 尝试提取字典中的评论
                                for key, value in item.items():
                                    if isinstance(value, str) and len(value) > 10:
                                        comments_list.append(f"{key}: {value}")
                                        break
                                else:
                                    comments_list.append(str(item))
                            else:
                                comments_list.append(str(item))

                        eval_data["comments"] = "\n\n".join(comments_list)
                    else:
                        # 其他类型直接转换为字符串
                        eval_data["comments"] = str(eval_data["comments"])
                except Exception as e:
                    logger.error(f"Error processing comments field: {e}")
                    eval_data["comments"] = "Error processing comments. The code may require manual review."

            # 清理comments中的控制字符
            if isinstance(eval_data["comments"], str):
                eval_data["comments"] = ''.join(ch for ch in eval_data["comments"]
                                              if ch >= ' ' or ch in '\t\n\r')

            logger.info(f"Successfully processed LLM response into evaluation data")
            return eval_data

        except Exception as e:
            logger.error(f"Failed to process LLM response: {e}")
            logger.error(f"Original response: {response_text[:500]}")

            # 返回默认评估数据
            return {
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
                "effective_additions": 0,
                "effective_deletions": 0,
                "estimated_hours": 0.5,
                "file_evaluations": {},  # 添加空的file_evaluations字段
                "comments": f"Failed to process evaluation data: {str(e)}. The code may require manual review."
            }


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
- effective_additions: (number of added lines that contain actual logic/functionality)
- effective_deletions: (number of deleted lines that contained actual logic/functionality)
- estimated_hours: (number of hours based primarily on effective changes)
- comments: (your detailed analysis including breakdown of effective vs non-effective changes)
- file_evaluations: (an object with file paths as keys, each containing individual evaluations with the same scoring fields)

CRITICAL: The fields effective_additions, effective_deletions, and file_evaluations are ABSOLUTELY REQUIRED and must be included in your response. Do not omit these fields under any circumstances.
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

            # generate eval_data from generated_text
            eval_data = process(generated_text)
            return eval_data
        except Exception as e:
            logger.error(f"Failed to extract JSON from response: {e}")
            return self._generate_default_scores(f"Failed to extract evaluation data: {str(e)}")

    def _estimate_default_hours(self, additions: int, deletions: int, file_path: str = None) -> float:
        try:
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
        except Exception as e:
            logger.error(f"Error estimating default hours: {e}")
            return 0.5  # Return a default of 0.5 hours if an error occurs

    async def evaluate_commit(
        self,
        commit_hash: str,
        commit_diff: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Evaluate a specific commit's changes.

        Args:
            commit_hash: The hash of the commit being evaluated
            commit_diff: Dictionary mapping file paths to their diffs and statistics

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

        # Always use whole commit evaluation
        logger.info(f"Using whole commit evaluation")
        print(f"Using whole commit evaluation (all files evaluated together)")

        # Evaluate the entire commit as a whole and extract individual file evaluations
        whole_commit_evaluation = await self.evaluate_commit_as_whole(commit_hash, commit_diff, extract_file_evaluations=True)
        print(f"DEBUG: Whole commit evaluation: {whole_commit_evaluation}")

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

            # Add effective additions and deletions
            evaluation_results["effective_additions"] = whole_commit_evaluation.get("effective_additions", total_additions)
            evaluation_results["effective_deletions"] = whole_commit_evaluation.get("effective_deletions", total_deletions)

            # Update statistics
            evaluation_results["statistics"]["total_effective_lines"] = whole_commit_evaluation.get("effective_code_lines", 0)
            evaluation_results["statistics"]["total_non_effective_lines"] = whole_commit_evaluation.get("non_effective_code_lines", 0)

            # Generate a summary
            evaluation_results["summary"] = whole_commit_evaluation.get("comments", "")[:200] + "..." if len(whole_commit_evaluation.get("comments", "")) > 200 else whole_commit_evaluation.get("comments", "")

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

    # 检查评估结果是否有效
    valid_results = [result for result in evaluation_results if hasattr(result, 'commit_evaluation') and result.commit_evaluation]
    if not valid_results:
        return "## 代码评价结果\n\n没有找到有效的评估结果。请检查日志文件了解详情。"

    # 添加LLM输出的完整JSON数据
    if len(evaluation_results) > 0 and hasattr(evaluation_results[0], 'commit_evaluation'):
        eval_result = evaluation_results[0].commit_evaluation
        if eval_result:
            logger.info(f"Adding complete LLM output to evaluation markdown")
            # 将在后面的代码中添加

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
        # 添加空值检查，如果evaluation为None，则使用默认值
        if eval is None:
            logger.warning(f"Evaluation is None for file {result.file_path} in commit {commit_hash}")
            # 使用默认值
            default_score = 5  # 默认分数为5
            commits_dict[commit_hash]["scores"]["readability"] += default_score
            commits_dict[commit_hash]["scores"]["efficiency"] += default_score
            commits_dict[commit_hash]["scores"]["security"] += default_score
            commits_dict[commit_hash]["scores"]["structure"] += default_score
            commits_dict[commit_hash]["scores"]["error_handling"] += default_score
            commits_dict[commit_hash]["scores"]["documentation"] += default_score
            commits_dict[commit_hash]["scores"]["code_style"] += default_score
            commits_dict[commit_hash]["scores"]["overall_score"] += default_score
        else:
            # 正常累加分数
            commits_dict[commit_hash]["scores"]["readability"] += eval.readability
            commits_dict[commit_hash]["scores"]["efficiency"] += eval.efficiency
            commits_dict[commit_hash]["scores"]["security"] += eval.security
            commits_dict[commit_hash]["scores"]["structure"] += eval.structure
            commits_dict[commit_hash]["scores"]["error_handling"] += eval.error_handling
            commits_dict[commit_hash]["scores"]["documentation"] += eval.documentation
            commits_dict[commit_hash]["scores"]["code_style"] += eval.code_style
            commits_dict[commit_hash]["scores"]["overall_score"] += eval.overall_score

        # 只计算一次工作时间（使用第一个文件的工作时间估算）
        if eval is not None and hasattr(eval, 'estimated_hours') and eval.estimated_hours and not commits_dict[commit_hash]["has_estimated_hours"]:
            commits_dict[commit_hash]["estimated_hours"] = eval.estimated_hours
            commits_dict[commit_hash]["has_estimated_hours"] = True

    # 按日期排序提交
    sorted_commits = sorted(commits_dict.values(), key=lambda x: x["date"])

    # 计算总体统计
    total_files = len(sorted_results)
    total_commits = len(sorted_commits)
    total_estimated_hours = 0  # 初始化总工作时间变量
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

    for commit in sorted_commits:
        file_count = len(commit["files"])
        # 计算每个提交的平均分
        for key in total_scores.keys():
            commit["scores"][key] = commit["scores"][key] / file_count
            total_scores[key] += commit["scores"][key]

    # 计算平均分
    avg_scores = {k: v / total_commits for k, v in total_scores.items()} if total_commits > 0 else {k: 0 for k in total_scores.keys()}

    # 添加概览
    markdown += f"## Overview\n\n"
    markdown += f"- **Developer**: {author}\n"
    markdown += f"- **Time Range**: {start_date} to {end_date}\n"
    markdown += f"- **Commits Evaluated**: {total_commits}\n"
    markdown += f"- **Files Evaluated**: {total_files}\n"

    # 添加LLM输出的完整JSON数据
    if len(evaluation_results) > 0 and hasattr(evaluation_results[0], 'commit_evaluation'):
        eval_result = evaluation_results[0].commit_evaluation
        if eval_result:
            markdown += f"\n## Complete LLM Evaluation Output\n\n"
            markdown += f"```json\n"
            if isinstance(eval_result, dict):
                # 使用json.dumps确保格式化良好
                import json
                try:
                    markdown += json.dumps(eval_result, indent=2, ensure_ascii=False)
                except Exception as e:
                    logger.error(f"Error formatting JSON: {e}")
                    markdown += str(eval_result)
            else:
                # 如果不是字典，尝试转换为字符串
                markdown += str(eval_result)
            markdown += f"\n```\n"

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

        # 不再显示单个提交的工作时间估算

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

            # 检查eval是否为None
            if eval is None:
                markdown += f"   - Score: N/A (evaluation not available)\n"
                markdown += f"   - Comments: No evaluation available\n\n"
            else:
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

            # 检查eval是否为None
            if eval is None:
                markdown += f"No evaluation available for this file.\n\n"
            else:
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
