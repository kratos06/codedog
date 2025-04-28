#!/usr/bin/env python
"""
Prompt Testing Tool for CodeDog using the original CodeDog prompts

This tool allows you to test code review prompts by providing a diff or code snippet
and getting the evaluation results using CodeDog's original prompts.
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
from codedog.utils.langchain_utils import load_model_by_name

# CodeDog的CODE_SUGGESTION提示模板
CODE_SUGGESTION = """Act as a senior code review expert with deep knowledge of industry standards and best practices for programming languages. I will give a code diff content.
Perform a comprehensive review of the code changes, conduct static analysis, and provide a detailed evaluation with specific scores based on the detailed criteria below.

## Review Requirements:
1. Provide a brief summary of the code's intended functionality and primary objectives
2. Conduct a thorough static analysis of code logic, performance, and security
3. Evaluate adherence to language-specific coding standards and best practices
4. Identify specific issues, vulnerabilities, and improvement opportunities
5. Score the code in each dimension using the detailed scoring criteria
6. Provide specific, actionable suggestions for improvement

## Language-Specific Standards:
{language} code should follow these standards:
- Use clear, descriptive variable and function names
- Follow consistent formatting and indentation
- Include appropriate comments and documentation
- Handle errors and edge cases properly
- Optimize for performance and resource usage
- Follow security best practices
- Adhere to language-specific conventions and idioms

### SCORES:
- Readability: [score] /10
- Efficiency & Performance: [score] /10
- Security: [score] /10
- Structure & Design: [score] /10
- Error Handling: [score] /10
- Documentation & Comments: [score] /10
- Code Style: [score] /10
- Final Overall Score: [calculated_overall_score] /10

Replace [score] with your actual numeric scores (e.g., 8.5).

Here's the code diff from file {name}:
```{language}
{content}
```

In addition to the code evaluation, please also estimate how many effective working hours an experienced programmer (5-10+ years) would need to complete these code changes. Include this estimate in your JSON response as 'estimated_hours'.

Please also analyze the code changes to determine how many lines are effective code changes (logic, functionality, algorithm changes) versus non-effective code changes (formatting, whitespace, comments, variable renaming without behavior changes). Include these counts in your JSON response as 'effective_code_lines' and 'non_effective_code_lines'.
"""


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


async def test_codedog_prompt(
    file_path: str,
    content: str,
    model_name: str = "gpt-3.5-turbo",
    output_format: str = "json"
) -> Dict[str, Any]:
    """
    使用CodeDog的提示测试代码评审

    Args:
        file_path: 文件路径
        content: 代码内容或差异内容
        model_name: 模型名称
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

    # 使用CodeDog的CODE_SUGGESTION提示
    prompt = CODE_SUGGESTION.format(
        language=language,
        name=file_path,
        content=sanitized_content
    )

    # 创建消息
    messages = [
        HumanMessage(content=prompt)
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
            # 尝试提取评分部分
            scores_section = re.search(r'### SCORES:\s*\n([\s\S]*?)(?:\n\n|\Z)', generated_text)
            if scores_section:
                scores_text = scores_section.group(1)
                scores_dict = {}

                # 提取各个评分
                for line in scores_text.split('\n'):
                    match = re.search(r'- ([\w\s&]+):\s*(\d+(\.\d+)?)\s*/10', line)
                    if match:
                        key = match.group(1).strip().lower().replace(' & ', '_').replace(' ', '_')
                        value = float(match.group(2))
                        scores_dict[key] = value

                # 提取评论
                comments_match = re.search(r'(?:Analysis|Comments|Suggestions):([\s\S]*?)(?=###|\Z)', generated_text, re.IGNORECASE)
                if comments_match:
                    scores_dict["comments"] = comments_match.group(1).strip()
                else:
                    scores_dict["comments"] = "No detailed comments provided."

                # 提取工作时间估算
                hours_match = re.search(r'(?:estimated_hours|working hours|time estimate).*?(\d+(\.\d+)?)', generated_text, re.IGNORECASE)
                if hours_match:
                    scores_dict["estimated_hours"] = float(hours_match.group(1))
                else:
                    scores_dict["estimated_hours"] = 0.0

                # 提取有效代码行数
                effective_match = re.search(r'(?:effective_code_lines|effective lines).*?(\d+)', generated_text, re.IGNORECASE)
                if effective_match:
                    scores_dict["effective_code_lines"] = int(effective_match.group(1))
                else:
                    scores_dict["effective_code_lines"] = 0

                # 提取非有效代码行数
                non_effective_match = re.search(r'(?:non_effective_code_lines|non-effective lines).*?(\d+)', generated_text, re.IGNORECASE)
                if non_effective_match:
                    scores_dict["non_effective_code_lines"] = int(non_effective_match.group(1))
                else:
                    scores_dict["non_effective_code_lines"] = 0

                result = scores_dict
                print("Successfully extracted scores from text")
            else:
                # 如果没有找到评分部分，尝试直接从文本中提取信息
                result = {
                    "comments": generated_text,
                    "summary": "See detailed analysis in comments."
                }

                # 尝试提取评分
                readability_match = re.search(r'readability.*?(\d+(\.\d+)?)', generated_text, re.IGNORECASE)
                if readability_match:
                    result["readability"] = float(readability_match.group(1))

                efficiency_match = re.search(r'efficiency.*?(\d+(\.\d+)?)', generated_text, re.IGNORECASE)
                if efficiency_match:
                    result["efficiency"] = float(efficiency_match.group(1))

                security_match = re.search(r'security.*?(\d+(\.\d+)?)', generated_text, re.IGNORECASE)
                if security_match:
                    result["security"] = float(security_match.group(1))

                structure_match = re.search(r'structure.*?(\d+(\.\d+)?)', generated_text, re.IGNORECASE)
                if structure_match:
                    result["structure"] = float(structure_match.group(1))

                error_handling_match = re.search(r'error handling.*?(\d+(\.\d+)?)', generated_text, re.IGNORECASE)
                if error_handling_match:
                    result["error_handling"] = float(error_handling_match.group(1))

                documentation_match = re.search(r'documentation.*?(\d+(\.\d+)?)', generated_text, re.IGNORECASE)
                if documentation_match:
                    result["documentation"] = float(documentation_match.group(1))

                code_style_match = re.search(r'code style.*?(\d+(\.\d+)?)', generated_text, re.IGNORECASE)
                if code_style_match:
                    result["code_style"] = float(code_style_match.group(1))

                overall_match = re.search(r'overall.*?(\d+(\.\d+)?)', generated_text, re.IGNORECASE)
                if overall_match:
                    result["overall_score"] = float(overall_match.group(1))

                # 提取工作时间估算
                hours_match = re.search(r'(?:estimated_hours|working hours|time estimate).*?(\d+(\.\d+)?)', generated_text, re.IGNORECASE)
                if hours_match:
                    result["estimated_hours"] = float(hours_match.group(1))
                else:
                    # 尝试找到任何数字后跟"hours"或"hour"的模式
                    hours_match = re.search(r'(\d+(\.\d+)?)\s*(?:hours?|hrs?)', generated_text, re.IGNORECASE)
                    if hours_match:
                        result["estimated_hours"] = float(hours_match.group(1))
                    else:
                        result["estimated_hours"] = 0.0

                # 提取有效代码行数
                effective_match = re.search(r'(?:effective_code_lines|effective lines).*?(\d+)', generated_text, re.IGNORECASE)
                if effective_match:
                    result["effective_code_lines"] = int(effective_match.group(1))
                else:
                    # 尝试找到任何数字后跟"effective"的模式
                    effective_match = re.search(r'(\d+)\s*(?:effective)', generated_text, re.IGNORECASE)
                    if effective_match:
                        result["effective_code_lines"] = int(effective_match.group(1))
                    else:
                        result["effective_code_lines"] = 0

                # 提取非有效代码行数
                non_effective_match = re.search(r'(?:non_effective_code_lines|non-effective lines).*?(\d+)', generated_text, re.IGNORECASE)
                if non_effective_match:
                    result["non_effective_code_lines"] = int(non_effective_match.group(1))
                else:
                    # 尝试找到任何数字后跟"non-effective"的模式
                    non_effective_match = re.search(r'(\d+)\s*(?:non-effective)', generated_text, re.IGNORECASE)
                    if non_effective_match:
                        result["non_effective_code_lines"] = int(non_effective_match.group(1))
                    else:
                        result["non_effective_code_lines"] = 0

                print("Extracted information directly from text")

    # 根据输出格式返回结果
    if output_format == "json":
        return result
    else:  # markdown
        # 将结果转换为Markdown格式
        markdown = f"# Code Review for {file_path}\n\n"
        markdown += f"## Scores\n\n"

        # 检查是否有static_analysis字段（CodeDog格式）
        if 'static_analysis' in result:
            static_analysis = result['static_analysis']
            markdown += f"- **Readability**: {static_analysis.get('readability', 'N/A')}/10\n"
            markdown += f"- **Efficiency & Performance**: {static_analysis.get('efficiency_performance', 'N/A')}/10\n"
            markdown += f"- **Security**: {static_analysis.get('security', 'N/A')}/10\n"
            markdown += f"- **Structure & Design**: {static_analysis.get('structure_design', 'N/A')}/10\n"
            markdown += f"- **Error Handling**: {static_analysis.get('error_handling', 'N/A')}/10\n"
            markdown += f"- **Documentation & Comments**: {static_analysis.get('documentation_comments', 'N/A')}/10\n"
            markdown += f"- **Code Style**: {static_analysis.get('code_style', 'N/A')}/10\n"
            markdown += f"- **Overall Score**: {static_analysis.get('overall_score', 'N/A')}/10\n\n"
        # 检查是否有scores字段（DeepSeek格式）
        elif 'scores' in result:
            scores = result['scores']
            markdown += f"- **Readability**: {scores.get('Readability', 'N/A')}/10\n"
            markdown += f"- **Efficiency & Performance**: {scores.get('Efficiency_Performance', 'N/A')}/10\n"
            markdown += f"- **Security**: {scores.get('Security', 'N/A')}/10\n"
            markdown += f"- **Structure & Design**: {scores.get('Structure_Design', 'N/A')}/10\n"
            markdown += f"- **Error Handling**: {scores.get('Error_Handling', 'N/A')}/10\n"
            markdown += f"- **Documentation & Comments**: {scores.get('Documentation_Comments', 'N/A')}/10\n"
            markdown += f"- **Code Style**: {scores.get('Code_Style', 'N/A')}/10\n"
            markdown += f"- **Overall Score**: {scores.get('Final_Overall_Score', 'N/A')}/10\n\n"
        # 检查是否有code_review.scores字段（DeepSeek格式）
        elif 'code_review' in result and 'scores' in result['code_review']:
            scores = result['code_review']['scores']
            markdown += f"- **Readability**: {scores.get('Readability', 'N/A')}/10\n"
            markdown += f"- **Efficiency & Performance**: {scores.get('Efficiency_Performance', 'N/A')}/10\n"
            markdown += f"- **Security**: {scores.get('Security', 'N/A')}/10\n"
            markdown += f"- **Structure & Design**: {scores.get('Structure_Design', 'N/A')}/10\n"
            markdown += f"- **Error Handling**: {scores.get('Error_Handling', 'N/A')}/10\n"
            markdown += f"- **Documentation & Comments**: {scores.get('Documentation_Comments', 'N/A')}/10\n"
            markdown += f"- **Code Style**: {scores.get('Code_Style', 'N/A')}/10\n"
            markdown += f"- **Overall Score**: {scores.get('Final_Overall_Score', 'N/A')}/10\n\n"
        else:
            # 直接从结果中获取评分
            markdown += f"- **Readability**: {result.get('readability', 'N/A')}/10\n"
            markdown += f"- **Efficiency & Performance**: {result.get('efficiency_&_performance', result.get('efficiency', 'N/A'))}/10\n"
            markdown += f"- **Security**: {result.get('security', 'N/A')}/10\n"
            markdown += f"- **Structure & Design**: {result.get('structure_&_design', result.get('structure', 'N/A'))}/10\n"
            markdown += f"- **Error Handling**: {result.get('error_handling', 'N/A')}/10\n"
            markdown += f"- **Documentation & Comments**: {result.get('documentation_&_comments', result.get('documentation', 'N/A'))}/10\n"
            markdown += f"- **Code Style**: {result.get('code_style', 'N/A')}/10\n"
            markdown += f"- **Overall Score**: {result.get('final_overall_score', result.get('overall_score', 'N/A'))}/10\n\n"

        markdown += f"## Code Change Analysis\n\n"

        # 检查是否有time_estimation字段（DeepSeek格式）
        if 'time_estimation' in result:
            time_estimation = result['time_estimation']
            markdown += f"- **Effective Code Lines**: {time_estimation.get('effective_code_lines', 'N/A')}\n"
            markdown += f"- **Non-Effective Code Lines**: {time_estimation.get('non_effective_code_lines', 'N/A')}\n"
            markdown += f"- **Estimated Hours**: {time_estimation.get('estimated_hours', 'N/A')}\n\n"
        # 检查是否有change_analysis字段（DeepSeek格式）
        elif 'change_analysis' in result:
            change_analysis = result['change_analysis']
            markdown += f"- **Effective Code Lines**: {change_analysis.get('effective_code_lines', 'N/A')}\n"
            markdown += f"- **Non-Effective Code Lines**: {change_analysis.get('non_effective_code_lines', 'N/A')}\n"
            markdown += f"- **Estimated Hours**: {change_analysis.get('estimated_hours', 'N/A')}\n\n"
        else:
            markdown += f"- **Effective Code Lines**: {result.get('effective_code_lines', 'N/A')}\n"
            markdown += f"- **Non-Effective Code Lines**: {result.get('non_effective_code_lines', 'N/A')}\n"
            markdown += f"- **Estimated Hours**: {result.get('estimated_hours', 'N/A')}\n\n"

        markdown += f"## Detailed Analysis\n\n"

        # 检查是否有DeepSeek格式的字段
        if 'review_summary' in result:
            review_summary = result['review_summary']
            markdown += f"**Intended Functionality**: {review_summary.get('intended_functionality', '')}\n\n"

            if 'primary_objectives' in review_summary:
                markdown += f"**Primary Objectives**:\n"
                for objective in review_summary.get('primary_objectives', []):
                    markdown += f"- {objective}\n"
                markdown += "\n"

        # 检查是否有code_review字段（DeepSeek格式）
        elif 'code_review' in result:
            code_review = result['code_review']

            if 'summary' in code_review:
                markdown += f"**Summary**: {code_review.get('summary', '')}\n\n"

            if 'static_analysis' in code_review:
                static_analysis = code_review['static_analysis']
                markdown += f"**Logic**: {static_analysis.get('logic', '')}\n\n"
                markdown += f"**Performance**: {static_analysis.get('performance', '')}\n\n"
                markdown += f"**Security**: {static_analysis.get('security', '')}\n\n"

            if 'issues_and_improvements' in code_review:
                markdown += f"**Issues and Improvements**:\n"
                for issue in code_review.get('issues_and_improvements', []):
                    markdown += f"- {issue}\n"
                markdown += "\n"

            if 'actionable_suggestions' in code_review:
                markdown += f"**Actionable Suggestions**:\n"
                for suggestion in code_review.get('actionable_suggestions', []):
                    markdown += f"- {suggestion}\n"
                markdown += "\n"

            if 'standards_adherence' in code_review:
                standards = code_review['standards_adherence']
                markdown += f"**Standards Adherence**:\n"
                markdown += f"- **Naming**: {standards.get('naming', '')}\n"
                markdown += f"- **Formatting**: {standards.get('formatting', '')}\n"
                markdown += f"- **Comments**: {standards.get('comments', '')}\n"
                markdown += f"- **Error Handling**: {standards.get('error_handling', '')}\n"
                markdown += f"- **Security Practices**: {standards.get('security_practices', '')}\n\n"

        if 'static_analysis' in result:
            static_analysis = result['static_analysis']
            markdown += f"**Logic**: {static_analysis.get('logic', '')}\n\n"
            markdown += f"**Performance**: {static_analysis.get('performance', '')}\n\n"
            markdown += f"**Security**: {static_analysis.get('security', '')}\n\n"

            if 'issues_identified' in static_analysis:
                markdown += f"**Issues Identified**:\n"
                for issue in static_analysis.get('issues_identified', []):
                    markdown += f"- {issue}\n"
                markdown += "\n"

            if 'improvement_opportunities' in static_analysis:
                markdown += f"**Improvement Opportunities**:\n"
                for opportunity in static_analysis.get('improvement_opportunities', []):
                    markdown += f"- {opportunity}\n"
                markdown += "\n"

        if 'actionable_suggestions' in result:
            markdown += f"**Actionable Suggestions**:\n"
            for suggestion in result.get('actionable_suggestions', []):
                markdown += f"- {suggestion}\n"
            markdown += "\n"

        if 'change_analysis' in result and 'breakdown' in result['change_analysis']:
            breakdown = result['change_analysis']['breakdown']

            if 'functional_changes' in breakdown:
                markdown += f"**Functional Changes**:\n"
                for change in breakdown.get('functional_changes', []):
                    markdown += f"- {change}\n"
                markdown += "\n"

            if 'non_functional_changes' in breakdown:
                markdown += f"**Non-Functional Changes**:\n"
                for change in breakdown.get('non_functional_changes', []):
                    markdown += f"- {change}\n"
                markdown += "\n"

        # 标准格式的字段
        if 'summary' in result:
            markdown += f"**Summary**: {result.get('summary', '')}\n\n"

        if 'suggestions' in result:
            markdown += f"**Suggestions**:\n"
            for suggestion in result.get('suggestions', []):
                markdown += f"- {suggestion}\n"
            markdown += "\n"

        markdown += result.get('comments', '')

        return {"markdown": markdown, "raw_result": result}


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Test CodeDog's code review prompts")

    # 输入选项
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--file", help="Path to the file to evaluate")
    input_group.add_argument("--diff", help="Path to the diff file to evaluate")

    # 模型选项
    parser.add_argument("--model", default="gpt-3.5-turbo", help="Model to use for evaluation (default: gpt-3.5-turbo)")

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

    # 测试提示
    result = await test_codedog_prompt(
        file_path=file_path,
        content=content,
        model_name=args.model,
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
