#!/usr/bin/env python
"""
Prompt Testing Tool for CodeDog

This tool allows you to test code review prompts by providing a diff or code snippet
and getting the evaluation results without running the full code review process.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Dict, Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# 设置日志记录
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

# 导入必要的模块
from codedog.utils.code_evaluator import DiffEvaluator
from codedog.utils.langchain_utils import load_model_by_name


def sanitize_content(content: str) -> str:
    """清理代码内容，移除异常字符"""
    # 移除不可打印字符，但保留换行符和制表符
    sanitized = ''.join(c for c in content if c.isprintable() or c in ['\n', '\t', '\r'])
    return sanitized


def guess_language(file_path: str) -> str:
    """根据文件扩展名猜测编程语言"""
    import os
    file_ext = os.path.splitext(file_path)[1].lower()
    
    # 文件扩展名到语言的映射
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
        
        # PHP
        '.php': 'PHP',
        
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
        '.css': 'CSS',
        '.scss': 'SCSS',
        '.sass': 'SASS',
        '.less': 'LESS',
        
        # Shell
        '.sh': 'Shell',
        '.bash': 'Shell',
        '.zsh': 'Shell',
        
        # SQL
        '.sql': 'SQL',
        
        # Markdown
        '.md': 'Markdown',
        '.markdown': 'Markdown',
        
        # JSON
        '.json': 'JSON',
        
        # YAML
        '.yml': 'YAML',
        '.yaml': 'YAML',
        
        # XML
        '.xml': 'XML',
        
        # Other
        '.txt': 'Text',
        '.csv': 'CSV',
    }
    
    return ext_to_lang.get(file_ext, 'Unknown')


async def test_prompt(
    file_path: str,
    content: str,
    model_name: str = "gpt-3.5-turbo",
    system_prompt: Optional[str] = None,
    output_format: str = "json"
) -> Dict[str, Any]:
    """
    测试代码评审提示
    
    Args:
        file_path: 文件路径
        content: 代码内容或差异内容
        model_name: 模型名称
        system_prompt: 系统提示，如果为None则使用默认系统提示
        output_format: 输出格式，可选值为json或markdown
        
    Returns:
        Dict[str, Any]: 评估结果
    """
    # 加载模型
    model = load_model_by_name(model_name)
    
    # 清理代码内容
    sanitized_content = sanitize_content(content)
    
    # 猜测语言
    language = guess_language(file_path)
    
    # 使用默认系统提示或自定义系统提示
    if system_prompt is None:
        system_prompt = """# ROLE AND OBJECTIVE
You are a senior code reviewer with 15+ years of experience across multiple programming languages and frameworks. Your task is to provide a thorough, objective evaluation of code quality and estimate the effort required to implement the changes.

# EVALUATION DIMENSIONS
Evaluate the code on these dimensions, scoring each from 1-10 (10 being highest):

1. Readability (1-10): Code clarity, naming conventions, consistent formatting
2. Efficiency (1-10): Algorithmic efficiency, resource usage, performance considerations
3. Security (1-10): Protection against vulnerabilities, input validation, secure coding practices
4. Structure (1-10): Architecture, modularity, separation of concerns, SOLID principles
5. Error Handling (1-10): Robust error handling, edge cases, graceful failure
6. Documentation (1-10): Comments, docstrings, self-documenting code
7. Code Style (1-10): Adherence to language-specific conventions and best practices
8. Overall Score (1-10): Comprehensive evaluation considering all dimensions

# CODE CHANGE CLASSIFICATION
When evaluating code changes (especially in diff format), carefully distinguish between:

## Non-Effective Changes (Should NOT count significantly toward working hours)
- Whitespace adjustments (spaces, tabs, line breaks)
- Indentation fixes without logic changes
- Comment additions or modifications without code changes
- Import reordering or reorganization
- Variable/function renaming without behavior changes
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
- For other languages: Apply relevant best practices and conventions"""
    
    # 创建用户提示
    user_prompt = f"""# Code Review Request

## File Information
- **File Name**: {file_path}
- **Language**: {language.lower()}

## Code to Review
```{language.lower()}
{sanitized_content}
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
    
    # 创建消息
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    # 调用模型
    print(f"Sending request to {model_name}...")
    start_time = time.time()
    response = await model.agenerate(messages=[messages])
    end_time = time.time()
    print(f"Response received in {end_time - start_time:.2f} seconds")
    
    # 获取响应文本
    generated_text = response.generations[0][0].text
    
    # 提取JSON
    try:
        # 尝试直接解析JSON
        result = json.loads(generated_text)
        print("Successfully parsed JSON response")
    except json.JSONDecodeError:
        # 如果直接解析失败，尝试提取JSON部分
        import re
        json_match = re.search(r'```json\s*(.*?)\s*```', generated_text, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group(1))
                print("Successfully extracted and parsed JSON from code block")
            except json.JSONDecodeError:
                print("Failed to parse JSON from code block")
                result = {
                    "error": "Failed to parse JSON response",
                    "raw_response": generated_text[:1000] + ("..." if len(generated_text) > 1000 else "")
                }
        else:
            print("No JSON code block found in response")
            result = {
                "error": "No JSON found in response",
                "raw_response": generated_text[:1000] + ("..." if len(generated_text) > 1000 else "")
            }
    
    # 根据输出格式返回结果
    if output_format == "json":
        return result
    else:  # markdown
        # 将结果转换为Markdown格式
        markdown = f"# Code Review for {file_path}\n\n"
        markdown += f"## Scores\n\n"
        markdown += f"- **Readability**: {result.get('readability', 'N/A')}/10\n"
        markdown += f"- **Efficiency**: {result.get('efficiency', 'N/A')}/10\n"
        markdown += f"- **Security**: {result.get('security', 'N/A')}/10\n"
        markdown += f"- **Structure**: {result.get('structure', 'N/A')}/10\n"
        markdown += f"- **Error Handling**: {result.get('error_handling', 'N/A')}/10\n"
        markdown += f"- **Documentation**: {result.get('documentation', 'N/A')}/10\n"
        markdown += f"- **Code Style**: {result.get('code_style', 'N/A')}/10\n"
        markdown += f"- **Overall Score**: {result.get('overall_score', 'N/A')}/10\n\n"
        
        markdown += f"## Code Change Analysis\n\n"
        markdown += f"- **Effective Code Lines**: {result.get('effective_code_lines', 'N/A')}\n"
        markdown += f"- **Non-Effective Code Lines**: {result.get('non_effective_code_lines', 'N/A')}\n"
        markdown += f"- **Estimated Hours**: {result.get('estimated_hours', 'N/A')}\n\n"
        
        markdown += f"## Detailed Analysis\n\n"
        markdown += result.get('comments', 'No comments provided')
        
        return {"markdown": markdown, "raw_result": result}


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Test code review prompts")
    
    # 输入选项
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--file", help="Path to the file to evaluate")
    input_group.add_argument("--diff", help="Path to the diff file to evaluate")
    
    # 模型选项
    parser.add_argument("--model", default="gpt-3.5-turbo", help="Model to use for evaluation (default: gpt-3.5-turbo)")
    
    # 系统提示选项
    parser.add_argument("--system-prompt", help="Path to a file containing a custom system prompt")
    
    # 输出选项
    parser.add_argument("--output", help="Path to save the output (default: stdout)")
    parser.add_argument("--format", choices=["json", "markdown"], default="json", help="Output format (default: json)")
    
    return parser.parse_args()


async def main():
    """主函数"""
    args = parse_args()
    
    # 读取输入内容
    if args.file:
        file_path = args.file
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    else:  # args.diff
        diff_path = args.diff
        with open(diff_path, "r", encoding="utf-8") as f:
            content = f.read()
        # 从diff文件名中提取原始文件名
        import os
        file_path = os.path.basename(diff_path)
        if file_path.endswith(".diff"):
            file_path = file_path[:-5]
    
    # 读取自定义系统提示
    system_prompt = None
    if args.system_prompt:
        with open(args.system_prompt, "r", encoding="utf-8") as f:
            system_prompt = f.read()
    
    # 测试提示
    result = await test_prompt(
        file_path=file_path,
        content=content,
        model_name=args.model,
        system_prompt=system_prompt,
        output_format=args.format
    )
    
    # 输出结果
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            if args.format == "json":
                json.dump(result, f, indent=2, ensure_ascii=False)
            else:  # markdown
                f.write(result["markdown"])
        print(f"Output saved to {args.output}")
    else:
        if args.format == "json":
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:  # markdown
            print(result["markdown"])


if __name__ == "__main__":
    asyncio.run(main())
