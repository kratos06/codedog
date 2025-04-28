#!/usr/bin/env python
"""
Batch Prompt Testing Tool for CodeDog

This script fetches multiple diffs from GitLab and tests code review prompts on them.
It allows comparing different prompts and models on real-world code changes.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# 设置日志记录
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

# 导入必要的模块
import gitlab
from test_prompt import test_prompt

# 创建输出目录
def create_output_dirs(base_dir: str) -> Tuple[str, str]:
    """创建输出目录结构"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"prompt_test_{timestamp}")
    diffs_dir = os.path.join(output_dir, "diffs")
    results_dir = os.path.join(output_dir, "results")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(diffs_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    return diffs_dir, results_dir

# 从GitLab获取MR的diff
def fetch_gitlab_diffs(
    project_id: str,
    mr_count: int = 5,
    max_files_per_mr: int = 3,
    include_extensions: Optional[List[str]] = None,
    exclude_extensions: Optional[List[str]] = None,
    state: str = "merged"
) -> List[Dict[str, Any]]:
    """
    从GitLab获取MR的diff
    
    Args:
        project_id: GitLab项目ID或路径
        mr_count: 要获取的MR数量
        max_files_per_mr: 每个MR最多获取的文件数
        include_extensions: 包含的文件扩展名列表
        exclude_extensions: 排除的文件扩展名列表
        state: MR状态，可选值为"merged", "opened", "closed"
        
    Returns:
        List[Dict[str, Any]]: MR的diff列表
    """
    # 获取GitLab配置
    gitlab_url = os.environ.get("GITLAB_URL", "https://gitlab.com")
    gitlab_token = os.environ.get("GITLAB_TOKEN")
    
    if not gitlab_token:
        raise ValueError("GitLab token not found in environment variables. Please set GITLAB_TOKEN.")
    
    # 连接GitLab
    gl = gitlab.Gitlab(gitlab_url, private_token=gitlab_token)
    
    # 获取项目
    try:
        project = gl.projects.get(project_id)
    except Exception as e:
        logger.error(f"Failed to get project {project_id}: {e}")
        raise
    
    logger.info(f"Successfully connected to GitLab project: {project.name}")
    
    # 获取MR列表
    mrs = project.mergerequests.list(state=state, order_by="updated_at", sort="desc", per_page=100)
    
    # 过滤MR
    filtered_mrs = []
    for mr in mrs[:mr_count]:
        logger.info(f"Processing MR #{mr.iid}: {mr.title}")
        
        # 获取MR的变更
        changes = mr.changes()
        
        # 过滤文件
        filtered_files = []
        for change in changes.get("changes", []):
            file_path = change.get("new_path") or change.get("old_path")
            
            # 检查文件扩展名
            _, ext = os.path.splitext(file_path)
            if include_extensions and ext.lower() not in include_extensions:
                continue
            if exclude_extensions and ext.lower() in exclude_extensions:
                continue
            
            # 获取diff
            diff = change.get("diff", "")
            
            # 添加到过滤后的文件列表
            filtered_files.append({
                "file_path": file_path,
                "diff": diff
            })
            
            # 如果达到每个MR的最大文件数，则停止
            if len(filtered_files) >= max_files_per_mr:
                break
        
        # 如果有过滤后的文件，则添加到MR列表
        if filtered_files:
            filtered_mrs.append({
                "id": mr.iid,
                "title": mr.title,
                "description": mr.description,
                "author": mr.author["name"],
                "created_at": mr.created_at,
                "updated_at": mr.updated_at,
                "files": filtered_files
            })
    
    logger.info(f"Fetched {len(filtered_mrs)} MRs with {sum(len(mr['files']) for mr in filtered_mrs)} files")
    return filtered_mrs

# 保存diff到文件
def save_diffs(mrs: List[Dict[str, Any]], diffs_dir: str) -> Dict[str, str]:
    """
    保存diff到文件
    
    Args:
        mrs: MR列表
        diffs_dir: diff文件保存目录
        
    Returns:
        Dict[str, str]: 文件路径到MR信息的映射
    """
    file_to_mr = {}
    
    for mr in mrs:
        mr_id = mr["id"]
        mr_title = mr["title"]
        
        for i, file_info in enumerate(mr["files"]):
            file_path = file_info["file_path"]
            diff = file_info["diff"]
            
            # 创建安全的文件名
            safe_name = f"mr_{mr_id}_{i}_{os.path.basename(file_path)}.diff"
            safe_path = os.path.join(diffs_dir, safe_name)
            
            # 保存diff
            with open(safe_path, "w", encoding="utf-8") as f:
                f.write(f"diff --git a/{file_path} b/{file_path}\n")
                f.write(diff)
            
            # 添加到映射
            file_to_mr[safe_path] = {
                "mr_id": mr_id,
                "mr_title": mr_title,
                "file_path": file_path
            }
            
            logger.info(f"Saved diff to {safe_path}")
    
    return file_to_mr

# 批量测试提示
async def batch_test(
    diff_files: Dict[str, Dict[str, Any]],
    results_dir: str,
    model_name: str = "gpt-3.5-turbo",
    system_prompt_path: Optional[str] = None,
    output_format: str = "json"
) -> Dict[str, Any]:
    """
    批量测试提示
    
    Args:
        diff_files: 文件路径到MR信息的映射
        results_dir: 结果保存目录
        model_name: 模型名称
        system_prompt_path: 系统提示文件路径
        output_format: 输出格式
        
    Returns:
        Dict[str, Any]: 测试结果
    """
    # 读取系统提示
    system_prompt = None
    if system_prompt_path:
        with open(system_prompt_path, "r", encoding="utf-8") as f:
            system_prompt = f.read()
    
    # 创建结果目录
    model_dir = os.path.join(results_dir, model_name.replace("-", "_"))
    os.makedirs(model_dir, exist_ok=True)
    
    # 创建汇总文件
    summary_file = os.path.join(model_dir, "summary.json")
    summary = {
        "model": model_name,
        "system_prompt": system_prompt_path,
        "timestamp": datetime.now().isoformat(),
        "results": {}
    }
    
    # 批量测试
    for diff_file, mr_info in diff_files.items():
        file_path = mr_info["file_path"]
        mr_id = mr_info["mr_id"]
        mr_title = mr_info["mr_title"]
        
        logger.info(f"Testing prompt on {os.path.basename(diff_file)} (MR #{mr_id}: {mr_title})")
        
        # 读取diff内容
        with open(diff_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        # 测试提示
        try:
            result = await test_prompt(
                file_path=file_path,
                content=content,
                model_name=model_name,
                system_prompt=system_prompt,
                output_format=output_format
            )
            
            # 保存结果
            result_file = os.path.join(model_dir, f"{os.path.basename(diff_file)}.{output_format}")
            with open(result_file, "w", encoding="utf-8") as f:
                if output_format == "json":
                    json.dump(result, f, indent=2, ensure_ascii=False)
                else:  # markdown
                    f.write(result["markdown"])
            
            # 添加到汇总
            summary["results"][os.path.basename(diff_file)] = {
                "mr_id": mr_id,
                "mr_title": mr_title,
                "file_path": file_path,
                "result_file": result_file,
                "scores": {
                    "readability": result.get("readability", "N/A"),
                    "efficiency": result.get("efficiency", "N/A"),
                    "security": result.get("security", "N/A"),
                    "structure": result.get("structure", "N/A"),
                    "error_handling": result.get("error_handling", "N/A"),
                    "documentation": result.get("documentation", "N/A"),
                    "code_style": result.get("code_style", "N/A"),
                    "overall_score": result.get("overall_score", "N/A"),
                    "effective_code_lines": result.get("effective_code_lines", "N/A"),
                    "non_effective_code_lines": result.get("non_effective_code_lines", "N/A"),
                    "estimated_hours": result.get("estimated_hours", "N/A")
                }
            }
            
            logger.info(f"Saved result to {result_file}")
            
        except Exception as e:
            logger.error(f"Error testing prompt on {diff_file}: {e}")
            # 添加错误信息到汇总
            summary["results"][os.path.basename(diff_file)] = {
                "mr_id": mr_id,
                "mr_title": mr_title,
                "file_path": file_path,
                "error": str(e)
            }
    
    # 保存汇总
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved summary to {summary_file}")
    
    return summary

# 生成比较报告
def generate_comparison_report(summaries: List[Dict[str, Any]], results_dir: str) -> str:
    """
    生成比较报告
    
    Args:
        summaries: 汇总列表
        results_dir: 结果保存目录
        
    Returns:
        str: 报告文件路径
    """
    # 创建报告文件
    report_file = os.path.join(results_dir, "comparison_report.md")
    
    # 生成报告
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# Prompt Testing Comparison Report\n\n")
        f.write(f"Generated at: {datetime.now().isoformat()}\n\n")
        
        # 模型信息
        f.write("## Models Tested\n\n")
        for i, summary in enumerate(summaries):
            model = summary["model"]
            system_prompt = summary["system_prompt"] or "Default"
            f.write(f"{i+1}. **{model}** with prompt: {system_prompt}\n")
        f.write("\n")
        
        # 汇总表格
        f.write("## Score Summary\n\n")
        f.write("| File | ")
        for summary in summaries:
            model = summary["model"]
            f.write(f"{model} Overall | ")
        f.write("\n")
        
        f.write("| --- | ")
        for _ in summaries:
            f.write("---: | ")
        f.write("\n")
        
        # 获取所有文件
        all_files = set()
        for summary in summaries:
            all_files.update(summary["results"].keys())
        
        # 填充表格
        for file in sorted(all_files):
            f.write(f"| {file} | ")
            for summary in summaries:
                result = summary["results"].get(file, {})
                if "error" in result:
                    f.write("Error | ")
                else:
                    overall_score = result.get("scores", {}).get("overall_score", "N/A")
                    f.write(f"{overall_score} | ")
            f.write("\n")
        
        # 详细比较
        f.write("\n## Detailed Comparison\n\n")
        for file in sorted(all_files):
            f.write(f"### {file}\n\n")
            
            # 创建比较表格
            f.write("| Metric | ")
            for summary in summaries:
                model = summary["model"]
                f.write(f"{model} | ")
            f.write("\n")
            
            f.write("| --- | ")
            for _ in summaries:
                f.write("---: | ")
            f.write("\n")
            
            # 填充表格
            metrics = ["readability", "efficiency", "security", "structure", 
                      "error_handling", "documentation", "code_style", "overall_score",
                      "effective_code_lines", "non_effective_code_lines", "estimated_hours"]
            
            for metric in metrics:
                metric_name = metric.replace("_", " ").title()
                f.write(f"| {metric_name} | ")
                for summary in summaries:
                    result = summary["results"].get(file, {})
                    if "error" in result:
                        f.write("Error | ")
                    else:
                        value = result.get("scores", {}).get(metric, "N/A")
                        f.write(f"{value} | ")
                f.write("\n")
            
            f.write("\n")
    
    logger.info(f"Generated comparison report: {report_file}")
    return report_file

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Batch test code review prompts on GitLab MRs")
    
    # GitLab选项
    parser.add_argument("--project", required=True, help="GitLab project ID or path")
    parser.add_argument("--mr-count", type=int, default=5, help="Number of MRs to fetch (default: 5)")
    parser.add_argument("--max-files", type=int, default=3, help="Maximum files per MR (default: 3)")
    parser.add_argument("--include", help="Included file extensions, comma separated, e.g. .py,.js")
    parser.add_argument("--exclude", help="Excluded file extensions, comma separated, e.g. .md,.txt")
    parser.add_argument("--state", choices=["merged", "opened", "closed"], default="merged", 
                      help="MR state to fetch (default: merged)")
    
    # 模型选项
    parser.add_argument("--models", help="Models to test, comma separated (default: gpt-3.5-turbo)")
    
    # 系统提示选项
    parser.add_argument("--system-prompts", help="Paths to system prompt files, comma separated")
    
    # 输出选项
    parser.add_argument("--output-dir", default="prompt_tests", help="Output directory (default: prompt_tests)")
    parser.add_argument("--format", choices=["json", "markdown"], default="json", help="Output format (default: json)")
    
    return parser.parse_args()

async def main():
    """主函数"""
    args = parse_args()
    
    # 解析包含和排除的文件扩展名
    include_extensions = None
    if args.include:
        include_extensions = [ext.strip() if ext.strip().startswith(".") else f".{ext.strip()}" 
                             for ext in args.include.split(",")]
    
    exclude_extensions = None
    if args.exclude:
        exclude_extensions = [ext.strip() if ext.strip().startswith(".") else f".{ext.strip()}" 
                             for ext in args.exclude.split(",")]
    
    # 解析模型列表
    models = ["gpt-3.5-turbo"]
    if args.models:
        models = [model.strip() for model in args.models.split(",")]
    
    # 解析系统提示列表
    system_prompts = [None]
    if args.system_prompts:
        system_prompts = [prompt.strip() for prompt in args.system_prompts.split(",")]
    
    # 创建输出目录
    diffs_dir, results_dir = create_output_dirs(args.output_dir)
    
    # 从GitLab获取MR的diff
    mrs = fetch_gitlab_diffs(
        project_id=args.project,
        mr_count=args.mr_count,
        max_files_per_mr=args.max_files,
        include_extensions=include_extensions,
        exclude_extensions=exclude_extensions,
        state=args.state
    )
    
    # 保存diff到文件
    diff_files = save_diffs(mrs, diffs_dir)
    
    # 批量测试提示
    summaries = []
    for model in models:
        for system_prompt in system_prompts:
            logger.info(f"Testing model {model} with prompt {system_prompt or 'Default'}")
            summary = await batch_test(
                diff_files=diff_files,
                results_dir=results_dir,
                model_name=model,
                system_prompt_path=system_prompt,
                output_format=args.format
            )
            summaries.append(summary)
    
    # 生成比较报告
    if len(summaries) > 1:
        report_file = generate_comparison_report(summaries, results_dir)
        print(f"\nComparison report generated: {report_file}")
    
    print(f"\nAll tests completed. Results saved to {results_dir}")

if __name__ == "__main__":
    asyncio.run(main())
