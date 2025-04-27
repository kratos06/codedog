import argparse
import asyncio
import time
import traceback
import logging
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional, Tuple
import os
import re
import sys
from datetime import datetime, timedelta

# Load environment variables from .env file
load_dotenv()

# Configure logger
logger = logging.getLogger(__name__)

from github import Github
from gitlab import Gitlab
from langchain_community.callbacks.manager import get_openai_callback

from codedog.actors.reporters.pull_request import PullRequestReporter
from codedog.chains import CodeReviewChain, PRSummaryChain
from codedog.retrievers import GithubRetriever, GitlabRetriever
from codedog.utils.langchain_utils import load_model_by_name
from codedog.utils.email_utils import send_report_email
from codedog.utils.git_hooks import install_git_hooks
from codedog.utils.git_log_analyzer import get_file_diffs_by_timeframe, get_commit_diff, CommitInfo
from codedog.utils.code_evaluator import DiffEvaluator, generate_evaluation_markdown


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="CodeDog - AI-powered code review tool")

    # Main operation subparsers
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # PR review command
    pr_parser = subparsers.add_parser("pr", help="Review a GitHub or GitLab pull request")
    pr_parser.add_argument("repository", help="Repository path (e.g. owner/repo)")
    pr_parser.add_argument("pr_number", type=int, help="Pull request number to review")
    pr_parser.add_argument("--platform", choices=["github", "gitlab"], default="github",
                         help="Platform to use (github or gitlab, defaults to github)")
    pr_parser.add_argument("--gitlab-url", help="GitLab URL (defaults to https://gitlab.com or GITLAB_URL env var)")
    pr_parser.add_argument("--email", help="Email addresses to send the report to (comma-separated)")

    # Setup git hooks command
    hook_parser = subparsers.add_parser("setup-hooks", help="Set up git hooks for commit-triggered reviews")
    hook_parser.add_argument("--repo", help="Path to git repository (defaults to current directory)")

    # Developer code evaluation command
    eval_parser = subparsers.add_parser("eval", help="Evaluate code commits of a developer in a time period")
    eval_parser.add_argument("author", help="Developer name or email (partial match)")
    eval_parser.add_argument("--start-date", help="Start date (YYYY-MM-DD), defaults to 7 days ago")
    eval_parser.add_argument("--end-date", help="End date (YYYY-MM-DD), defaults to today")
    eval_parser.add_argument("--repo", help="Git repository path or name (e.g. owner/repo for remote repositories)")
    eval_parser.add_argument("--include", help="Included file extensions, comma separated, e.g. .py,.js")
    eval_parser.add_argument("--exclude", help="Excluded file extensions, comma separated, e.g. .md,.txt")
    eval_parser.add_argument("--model", help="Evaluation model, defaults to CODE_REVIEW_MODEL env var or gpt-3.5")
    eval_parser.add_argument("--email", help="Email addresses to send the report to (comma-separated)")
    eval_parser.add_argument("--output", help="Report output path, defaults to codedog_eval_<author>_<date>.md")
    eval_parser.add_argument("--platform", choices=["github", "gitlab", "local"], default="local",
                         help="Platform to use (github, gitlab, or local, defaults to local)")
    eval_parser.add_argument("--gitlab-url", help="GitLab URL (defaults to https://gitlab.com or GITLAB_URL env var)")

    # Commit review command
    commit_parser = subparsers.add_parser("commit", help="Review a specific commit")
    commit_parser.add_argument("commit_hash", help="Commit hash to review")
    commit_parser.add_argument("--repo", help="Git repository path or name (e.g. owner/repo for remote repositories)")
    commit_parser.add_argument("--include", help="Included file extensions, comma separated, e.g. .py,.js")
    commit_parser.add_argument("--exclude", help="Excluded file extensions, comma separated, e.g. .md,.txt")
    commit_parser.add_argument("--model", help="Review model, defaults to CODE_REVIEW_MODEL env var or gpt-3.5")
    commit_parser.add_argument("--email", help="Email addresses to send the report to (comma-separated)")
    commit_parser.add_argument("--output", help="Report output path, defaults to codedog_commit_<hash>_<date>.md")
    commit_parser.add_argument("--platform", choices=["github", "gitlab", "local"], default="local",
                         help="Platform to use (github, gitlab, or local, defaults to local)")
    commit_parser.add_argument("--gitlab-url", help="GitLab URL (defaults to https://gitlab.com or GITLAB_URL env var)")

    # Repository evaluation command
    repo_eval_parser = subparsers.add_parser("repo-eval", help="Evaluate all commits in a repository within a time period for all committers")
    repo_eval_parser.add_argument("repo", help="Git repository path or name (e.g. owner/repo for remote repositories)")
    repo_eval_parser.add_argument("--start-date", help="Start date (YYYY-MM-DD), defaults to 7 days ago")
    repo_eval_parser.add_argument("--end-date", help="End date (YYYY-MM-DD), defaults to today")
    repo_eval_parser.add_argument("--include", help="Included file extensions, comma separated, e.g. .py,.js")
    repo_eval_parser.add_argument("--exclude", help="Excluded file extensions, comma separated, e.g. .md,.txt")
    repo_eval_parser.add_argument("--model", help="Evaluation model, defaults to CODE_REVIEW_MODEL env var or gpt-3.5")
    repo_eval_parser.add_argument("--email", help="Email addresses to send the report to (comma-separated)")
    repo_eval_parser.add_argument("--output-dir", help="Directory to save reports, defaults to codedog_repo_eval_<date>")
    repo_eval_parser.add_argument("--platform", choices=["github", "gitlab", "local"], default="local",
                         help="Platform to use (github, gitlab, or local, defaults to local)")
    repo_eval_parser.add_argument("--gitlab-url", help="GitLab URL (defaults to https://gitlab.com or GITLAB_URL env var)")

    return parser.parse_args()


def parse_emails(emails_str: Optional[str]) -> List[str]:
    """Parse comma-separated email addresses."""
    if not emails_str:
        return []

    return [email.strip() for email in emails_str.split(",") if email.strip()]


def parse_extensions(extensions_str: Optional[str]) -> Optional[List[str]]:
    """Parse comma-separated file extensions."""
    if not extensions_str:
        return None

    return [ext.strip() for ext in extensions_str.split(",") if ext.strip()]


async def pr_summary(retriever, summary_chain):
    """Generate PR summary asynchronously."""
    result = await summary_chain.ainvoke(
        {"pull_request": retriever.pull_request}, include_run_info=True
    )
    return result


async def code_review(retriever, review_chain):
    """Generate code review asynchronously."""
    result = await review_chain.ainvoke(
        {"pull_request": retriever.pull_request}, include_run_info=True
    )
    return result


def get_remote_commit_diff(
    platform: str,
    repository_name: str,
    commit_hash: str,
    include_extensions: Optional[List[str]] = None,
    exclude_extensions: Optional[List[str]] = None,
    gitlab_url: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Get commit diff from remote repositories (GitHub or GitLab).

    Args:
        platform (str): Platform to use (github or gitlab)
        repository_name (str): Repository name (e.g. owner/repo)
        commit_hash (str): Commit hash to review
        include_extensions (Optional[List[str]], optional): File extensions to include. Defaults to None.
        exclude_extensions (Optional[List[str]], optional): File extensions to exclude. Defaults to None.
        gitlab_url (Optional[str], optional): GitLab URL. Defaults to None.

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary mapping file paths to their diffs and statistics
    """
    logger.info(f"Getting commit diff from {platform} for repository {repository_name}, commit {commit_hash}")
    logger.info(f"Include extensions: {include_extensions}, Exclude extensions: {exclude_extensions}")

    if platform.lower() == "github":
        # Initialize GitHub client
        github_token = os.environ.get("GITHUB_TOKEN", "")
        if not github_token:
            error_msg = "GITHUB_TOKEN environment variable is not set"
            logger.error(error_msg)
            print(error_msg)
            return {}

        github_client = Github(github_token)
        print(f"Analyzing GitHub repository {repository_name} for commit {commit_hash}")
        logger.info(f"Initialized GitHub client for repository {repository_name}")

        try:
            # Get repository
            logger.info(f"Fetching repository {repository_name}")
            repo = github_client.get_repo(repository_name)

            # Get commit
            logger.info(f"Fetching commit {commit_hash}")
            commit = repo.get_commit(commit_hash)
            logger.info(f"Commit found: {commit.sha}, author: {commit.commit.author.name}, date: {commit.commit.author.date}")

            # Extract file diffs
            file_diffs = {}
            logger.info(f"Processing {len(commit.files)} files in commit")

            for i, file in enumerate(commit.files):
                logger.info(f"Processing file {i+1}/{len(commit.files)}: {file.filename}")

                # Filter by file extensions
                _, ext = os.path.splitext(file.filename)
                logger.debug(f"File extension: {ext}")

                if include_extensions and ext not in include_extensions:
                    logger.info(f"Skipping file {file.filename} - extension {ext} not in include list")
                    continue
                if exclude_extensions and ext in exclude_extensions:
                    logger.info(f"Skipping file {file.filename} - extension {ext} in exclude list")
                    continue

                if file.patch:
                    logger.info(f"Adding file {file.filename} to diff (status: {file.status}, additions: {file.additions}, deletions: {file.deletions})")
                    file_diffs[file.filename] = {
                        "diff": f"diff --git a/{file.filename} b/{file.filename}\n{file.patch}",
                        "status": file.status,
                        "additions": file.additions,
                        "deletions": file.deletions,
                    }
                else:
                    logger.warning(f"No patch content for file {file.filename}")

            logger.info(f"Processed {len(file_diffs)} files after filtering")
            return file_diffs

        except Exception as e:
            error_msg = f"Failed to retrieve GitHub commit: {str(e)}"
            logger.error(error_msg, exc_info=True)
            print(error_msg)
            return {}

    elif platform.lower() == "gitlab":
        # Initialize GitLab client
        gitlab_token = os.environ.get("GITLAB_TOKEN", "")
        if not gitlab_token:
            error_msg = "GITLAB_TOKEN environment variable is not set"
            logger.error(error_msg)
            print(error_msg)
            return {}

        # Use provided GitLab URL or fall back to environment variable or default
        gitlab_url = gitlab_url or os.environ.get("GITLAB_URL", "https://gitlab.com")
        logger.info(f"Using GitLab URL: {gitlab_url}")

        gitlab_client = Gitlab(url=gitlab_url, private_token=gitlab_token)
        print(f"Analyzing GitLab repository {repository_name} for commit {commit_hash}")
        logger.info(f"Initialized GitLab client for repository {repository_name}")

        try:
            # Get repository
            logger.info(f"Fetching project {repository_name}")
            project = gitlab_client.projects.get(repository_name)
            logger.info(f"Project found: {project.name}, ID: {project.id}")

            # Get commit
            logger.info(f"Fetching commit {commit_hash}")
            commit = project.commits.get(commit_hash)
            logger.info(f"Commit found: {commit.id}, author: {commit.author_name}, date: {commit.created_at}")

            # Get commit diff
            logger.info("Fetching commit diff")
            diff = commit.diff()
            logger.info(f"Processing {len(diff)} files in commit diff")

            # Extract file diffs
            file_diffs = {}
            for i, file_diff in enumerate(diff):
                file_path = file_diff.get('new_path', '')
                old_path = file_diff.get('old_path', '')
                diff_content = file_diff.get('diff', '')

                logger.info(f"Processing file {i+1}/{len(diff)}: {file_path}")
                logger.debug(f"Old path: {old_path}, New path: {file_path}")

                # Skip if no diff content
                if not diff_content:
                    logger.warning(f"No diff content for file {file_path}, skipping")
                    continue

                # Filter by file extensions
                _, ext = os.path.splitext(file_path)
                logger.debug(f"File extension: {ext}")

                if include_extensions and ext not in include_extensions:
                    logger.info(f"Skipping file {file_path} - extension {ext} not in include list")
                    continue
                if exclude_extensions and ext in exclude_extensions:
                    logger.info(f"Skipping file {file_path} - extension {ext} in exclude list")
                    continue

                # Determine file status
                if file_diff.get('new_file', False):
                    status = 'A'  # Added
                elif file_diff.get('deleted_file', False):
                    status = 'D'  # Deleted
                else:
                    status = 'M'  # Modified

                logger.debug(f"File status: {status}")

                # Format diff content
                formatted_diff = f"diff --git a/{old_path} b/{file_path}\n{diff_content}"

                # Count additions and deletions
                additions = diff_content.count('\n+')
                deletions = diff_content.count('\n-')
                logger.debug(f"Additions: {additions}, Deletions: {deletions}")

                logger.info(f"Adding file {file_path} to diff (status: {status}, additions: {additions}, deletions: {deletions})")
                file_diffs[file_path] = {
                    "diff": formatted_diff,
                    "status": status,
                    "additions": additions,
                    "deletions": deletions,
                }

            logger.info(f"Processed {len(file_diffs)} files after filtering")
            return file_diffs

        except Exception as e:
            error_msg = f"Failed to retrieve GitLab commit: {str(e)}"
            logger.error(error_msg, exc_info=True)
            print(error_msg)
            return {}

    else:
        error_msg = f"Unsupported platform: {platform}. Use 'github' or 'gitlab'."
        logger.error(error_msg)
        print(error_msg)
        return {}


def get_all_remote_commits(
    platform: str,
    repository_name: str,
    start_date: str,
    end_date: str,
    include_extensions: Optional[List[str]] = None,
    exclude_extensions: Optional[List[str]] = None,
    gitlab_url: Optional[str] = None,
) -> Dict[str, Tuple[List[Any], Dict[str, Dict[str, str]], Dict[str, int]]]:
    """
    Get all commits from remote repositories (GitHub or GitLab) grouped by author.

    Args:
        platform (str): Platform to use (github or gitlab)
        repository_name (str): Repository name (e.g. owner/repo)
        start_date (str): Start date (YYYY-MM-DD)
        end_date (str): End date (YYYY-MM-DD)
        include_extensions (Optional[List[str]], optional): File extensions to include. Defaults to None.
        exclude_extensions (Optional[List[str]], optional): File extensions to exclude. Defaults to None.
        gitlab_url (Optional[str], optional): GitLab URL. Defaults to None.

    Returns:
        Dict[str, Tuple[List[Any], Dict[str, Dict[str, str]], Dict[str, int]]]: Dictionary mapping author names to their commits, file diffs, and code stats
    """
    if platform.lower() == "github":
        # Initialize GitHub client
        github_token = os.environ.get("GITHUB_TOKEN", "")
        if not github_token:
            error_msg = "GITHUB_TOKEN environment variable is not set"
            logger.error(error_msg)
            print(error_msg)
            return {}

        github_client = Github(github_token)
        print(f"Analyzing GitHub repository {repository_name} for all commits")
        logger.info(f"Initialized GitHub client for repository {repository_name}")

        try:
            # Get repository
            repo = github_client.get_repo(repository_name)

            # Convert dates to datetime objects
            start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
            end_datetime = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)  # Include the end date

            # Get all commits in the repository within the date range
            all_commits = repo.get_commits(since=start_datetime, until=end_datetime)

            # Group commits by author
            author_commits = {}

            for commit in all_commits:
                author_name = commit.commit.author.name
                author_email = commit.commit.author.email

                # Use email as part of the key to distinguish between authors with the same name
                author_key = f"{author_name} <{author_email}>" if author_email else author_name

                if author_key not in author_commits:
                    author_commits[author_key] = {
                        "commits": [],
                        "file_diffs": {},
                        "stats": {
                            "total_added_lines": 0,
                            "total_deleted_lines": 0,
                            "total_effective_lines": 0,
                            "total_files": set()
                        }
                    }

                # Create CommitInfo object
                commit_info = CommitInfo(
                    hash=commit.sha,
                    author=author_name,
                    date=commit.commit.author.date,
                    message=commit.commit.message,
                    files=[file.filename for file in commit.files],
                    diff="\n".join([f"diff --git a/{file.filename} b/{file.filename}\n{file.patch}" for file in commit.files if file.patch]),
                    added_lines=sum(file.additions for file in commit.files),
                    deleted_lines=sum(file.deletions for file in commit.files),
                    effective_lines=sum(file.additions - file.deletions for file in commit.files)
                )

                author_commits[author_key]["commits"].append(commit_info)

                # Extract file diffs
                file_diffs = {}
                for file in commit.files:
                    if file.patch:
                        # Filter by file extensions
                        _, ext = os.path.splitext(file.filename)
                        if include_extensions and ext not in include_extensions:
                            continue
                        if exclude_extensions and ext in exclude_extensions:
                            continue

                        file_diffs[file.filename] = file.patch
                        author_commits[author_key]["stats"]["total_files"].add(file.filename)

                author_commits[author_key]["file_diffs"][commit.sha] = file_diffs

                # Update stats
                author_commits[author_key]["stats"]["total_added_lines"] += commit_info.added_lines
                author_commits[author_key]["stats"]["total_deleted_lines"] += commit_info.deleted_lines
                author_commits[author_key]["stats"]["total_effective_lines"] += commit_info.effective_lines

            # Convert the set of files to count
            for author_key in author_commits:
                author_commits[author_key]["stats"]["total_files"] = len(author_commits[author_key]["stats"]["total_files"])

            # Convert to the expected return format
            result = {}
            for author_key, data in author_commits.items():
                result[author_key] = (data["commits"], data["file_diffs"], data["stats"])

            return result

        except Exception as e:
            error_msg = f"Failed to retrieve GitHub commits: {str(e)}"
            logger.error(error_msg, exc_info=True)
            print(error_msg)
            return {}

    elif platform.lower() == "gitlab":
        # Initialize GitLab client
        gitlab_token = os.environ.get("GITLAB_TOKEN", "")
        if not gitlab_token:
            error_msg = "GITLAB_TOKEN environment variable is not set"
            logger.error(error_msg)
            print(error_msg)
            return {}

        # Use provided GitLab URL or fall back to environment variable or default
        gitlab_url = gitlab_url or os.environ.get("GITLAB_URL", "https://gitlab.com")
        logger.info(f"Using GitLab URL: {gitlab_url}")

        gitlab_client = Gitlab(url=gitlab_url, private_token=gitlab_token)
        print(f"Analyzing GitLab repository {repository_name} for all commits")
        logger.info(f"Initialized GitLab client for repository {repository_name}")

        try:
            # Get repository
            project = gitlab_client.projects.get(repository_name)
            logger.info(f"Project found: {project.name}, ID: {project.id}")

            # Convert dates to ISO format
            start_iso = f"{start_date}T00:00:00Z"
            end_iso = f"{end_date}T23:59:59Z"

            # Get all commits in the repository within the date range
            all_commits = project.commits.list(all=True, since=start_iso, until=end_iso)
            logger.info(f"Found {len(all_commits)} commits in the date range")

            # Group commits by author
            author_commits = {}

            for commit in all_commits:
                author_name = commit.author_name
                author_email = commit.author_email

                # Use email as part of the key to distinguish between authors with the same name
                author_key = f"{author_name} <{author_email}>" if author_email else author_name

                if author_key not in author_commits:
                    author_commits[author_key] = {
                        "commits": [],
                        "file_diffs": {},
                        "stats": {
                            "total_added_lines": 0,
                            "total_deleted_lines": 0,
                            "total_effective_lines": 0,
                            "total_files": set()
                        }
                    }

                # Get commit details
                commit_detail = project.commits.get(commit.id)

                # Get commit diff
                diff = commit_detail.diff()

                # Filter files by extension
                filtered_diff = []
                for file_diff in diff:
                    file_path = file_diff.get('new_path', '')
                    _, ext = os.path.splitext(file_path)

                    if include_extensions and ext not in include_extensions:
                        continue
                    if exclude_extensions and ext in exclude_extensions:
                        continue

                    filtered_diff.append(file_diff)

                # Skip if no files match the filter
                if not filtered_diff:
                    continue

                # Get file content for each modified file
                file_diffs = {}
                for file_diff in filtered_diff:
                    file_path = file_diff.get('new_path', '')
                    old_path = file_diff.get('old_path', '')
                    diff_content = file_diff.get('diff', '')

                    # Skip if no diff content
                    if not diff_content:
                        continue

                    # Format diff content
                    formatted_diff = f"diff --git a/{old_path} b/{file_path}\n{diff_content}"
                    file_diffs[file_path] = formatted_diff
                    author_commits[author_key]["stats"]["total_files"].add(file_path)

                # Skip if no valid diffs
                if not file_diffs:
                    continue

                # Count additions and deletions
                added_lines = sum(diff_content.count('\n+') for diff_content in file_diffs.values())
                deleted_lines = sum(diff_content.count('\n-') for diff_content in file_diffs.values())
                effective_lines = added_lines - deleted_lines

                # Create CommitInfo object
                commit_info = CommitInfo(
                    hash=commit.id,
                    author=author_name,
                    date=datetime.strptime(commit.created_at, "%Y-%m-%dT%H:%M:%S.%f%z") if '.' in commit.created_at else datetime.strptime(commit.created_at, "%Y-%m-%dT%H:%M:%SZ"),
                    message=commit.message,
                    files=list(file_diffs.keys()),
                    diff="\n\n".join(file_diffs.values()),
                    added_lines=added_lines,
                    deleted_lines=deleted_lines,
                    effective_lines=effective_lines
                )

                author_commits[author_key]["commits"].append(commit_info)
                author_commits[author_key]["file_diffs"][commit.id] = file_diffs

                # Update stats
                author_commits[author_key]["stats"]["total_added_lines"] += added_lines
                author_commits[author_key]["stats"]["total_deleted_lines"] += deleted_lines
                author_commits[author_key]["stats"]["total_effective_lines"] += effective_lines

            # Convert the set of files to count
            for author_key in author_commits:
                author_commits[author_key]["stats"]["total_files"] = len(author_commits[author_key]["stats"]["total_files"])

            # Convert to the expected return format
            result = {}
            for author_key, data in author_commits.items():
                result[author_key] = (data["commits"], data["file_diffs"], data["stats"])

            return result

        except Exception as e:
            error_msg = f"Failed to retrieve GitLab commits: {str(e)}"
            logger.error(error_msg, exc_info=True)
            print(error_msg)
            return {}

    else:
        error_msg = f"Unsupported platform: {platform}. Use 'github' or 'gitlab'."
        logger.error(error_msg)
        print(error_msg)
        return {}

def get_remote_commits(
    platform: str,
    repository_name: str,
    author: str,
    start_date: str,
    end_date: str,
    include_extensions: Optional[List[str]] = None,
    exclude_extensions: Optional[List[str]] = None,
    gitlab_url: Optional[str] = None,
) -> Tuple[List[Any], Dict[str, Dict[str, str]], Dict[str, int]]:
    """
    Get commits from remote repositories (GitHub or GitLab).

    Args:
        platform (str): Platform to use (github or gitlab)
        repository_name (str): Repository name (e.g. owner/repo)
        author (str): Author name or email
        start_date (str): Start date (YYYY-MM-DD)
        end_date (str): End date (YYYY-MM-DD)
        include_extensions (Optional[List[str]], optional): File extensions to include. Defaults to None.
        exclude_extensions (Optional[List[str]], optional): File extensions to exclude. Defaults to None.
        gitlab_url (Optional[str], optional): GitLab URL. Defaults to None.

    Returns:
        Tuple[List[Any], Dict[str, Dict[str, str]], Dict[str, int]]: Commits, file diffs, and code stats
    """
    if platform.lower() == "github":
        # Initialize GitHub client
        github_client = Github()  # Will automatically load GITHUB_TOKEN from environment
        print(f"Analyzing GitHub repository {repository_name} for commits by {author}")

        try:
            # Get repository
            repo = github_client.get_repo(repository_name)

            # Convert dates to datetime objects
            start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
            end_datetime = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)  # Include the end date

            # Get commits
            commits = []
            commit_file_diffs = {}

            # Get all commits in the repository within the date range
            all_commits = repo.get_commits(since=start_datetime, until=end_datetime)

            # Filter by author
            for commit in all_commits:
                if author.lower() in commit.commit.author.name.lower() or (
                    commit.commit.author.email and author.lower() in commit.commit.author.email.lower()
                ):
                    # Create CommitInfo object
                    commit_info = CommitInfo(
                        hash=commit.sha,
                        author=commit.commit.author.name,
                        date=commit.commit.author.date,
                        message=commit.commit.message,
                        files=[file.filename for file in commit.files],
                        diff="\n".join([f"diff --git a/{file.filename} b/{file.filename}\n{file.patch}" for file in commit.files if file.patch]),
                        added_lines=sum(file.additions for file in commit.files),
                        deleted_lines=sum(file.deletions for file in commit.files),
                        effective_lines=sum(file.additions - file.deletions for file in commit.files)
                    )
                    commits.append(commit_info)

                    # Extract file diffs
                    file_diffs = {}
                    for file in commit.files:
                        if file.patch:
                            # Filter by file extensions
                            _, ext = os.path.splitext(file.filename)
                            if include_extensions and ext not in include_extensions:
                                continue
                            if exclude_extensions and ext in exclude_extensions:
                                continue

                            file_diffs[file.filename] = file.patch

                    commit_file_diffs[commit.sha] = file_diffs

            # Calculate code stats
            code_stats = {
                "total_added_lines": sum(commit.added_lines for commit in commits),
                "total_deleted_lines": sum(commit.deleted_lines for commit in commits),
                "total_effective_lines": sum(commit.effective_lines for commit in commits),
                "total_files": len(set(file for commit in commits for file in commit.files))
            }

            return commits, commit_file_diffs, code_stats

        except Exception as e:
            error_msg = f"Failed to retrieve GitHub commits: {str(e)}"
            print(error_msg)
            return [], {}, {}

    elif platform.lower() == "gitlab":
        # Initialize GitLab client
        gitlab_token = os.environ.get("GITLAB_TOKEN", "")
        if not gitlab_token:
            error_msg = "GITLAB_TOKEN environment variable is not set"
            print(error_msg)
            return [], {}, {}

        # Use provided GitLab URL or fall back to environment variable or default
        gitlab_url = gitlab_url or os.environ.get("GITLAB_URL", "https://gitlab.com")

        gitlab_client = Gitlab(url=gitlab_url, private_token=gitlab_token)
        print(f"Analyzing GitLab repository {repository_name} for commits by {author}")

        try:
            # Get repository
            project = gitlab_client.projects.get(repository_name)

            # Get commits
            commits = []
            commit_file_diffs = {}

            # Convert dates to ISO format
            start_iso = f"{start_date}T00:00:00Z"
            end_iso = f"{end_date}T23:59:59Z"

            # Get all commits in the repository within the date range
            all_commits = project.commits.list(all=True, since=start_iso, until=end_iso)

            # Filter by author
            for commit in all_commits:
                if author.lower() in commit.author_name.lower() or (
                    commit.author_email and author.lower() in commit.author_email.lower()
                ):
                    # Get commit details
                    commit_detail = project.commits.get(commit.id)

                    # Get commit diff
                    diff = commit_detail.diff()

                    # Filter files by extension
                    filtered_diff = []
                    for file_diff in diff:
                        file_path = file_diff.get('new_path', '')
                        _, ext = os.path.splitext(file_path)

                        if include_extensions and ext not in include_extensions:
                            continue
                        if exclude_extensions and ext in exclude_extensions:
                            continue

                        filtered_diff.append(file_diff)

                    # Skip if no files match the filter
                    if not filtered_diff:
                        continue

                    # Get file content for each modified file
                    file_diffs = {}
                    for file_diff in filtered_diff:
                        file_path = file_diff.get('new_path', '')
                        old_path = file_diff.get('old_path', '')
                        diff_content = file_diff.get('diff', '')

                        # Skip if no diff content
                        if not diff_content:
                            continue

                        # Try to get the file content
                        try:
                            # For new files, get the content from the current commit
                            if file_diff.get('new_file', False):
                                try:
                                    # Get the file content and handle both string and bytes
                                    file_obj = project.files.get(file_path=file_path, ref=commit.id)
                                    if hasattr(file_obj, 'content'):
                                        # Raw content from API
                                        file_content = file_obj.content
                                    elif hasattr(file_obj, 'decode'):
                                        # Decode if it's bytes
                                        try:
                                            file_content = file_obj.decode()
                                        except TypeError:
                                            # If decode fails, try to get content directly
                                            file_content = file_obj.content if hasattr(file_obj, 'content') else str(file_obj)
                                    else:
                                        # Fallback to string representation
                                        file_content = str(file_obj)

                                    # Format as a proper diff with the entire file as added
                                    formatted_diff = f"diff --git a/{old_path} b/{file_path}\n--- /dev/null\n+++ b/{file_path}\n"
                                    formatted_diff += "\n".join([f"+{line}" for line in file_content.split('\n')])
                                    file_diffs[file_path] = formatted_diff
                                except Exception as e:
                                    print(f"Warning: Could not get content for new file {file_path}: {str(e)}")
                                    # Try to get the raw file content directly from the API
                                    try:
                                        import base64
                                        raw_file = project.repository_files.get(file_path=file_path, ref=commit.id)
                                        if raw_file and hasattr(raw_file, 'content'):
                                            # Decode base64 content if available
                                            try:
                                                decoded_content = base64.b64decode(raw_file.content).decode('utf-8', errors='replace')
                                                formatted_diff = f"diff --git a/{old_path} b/{file_path}\n--- /dev/null\n+++ b/{file_path}\n"
                                                formatted_diff += "\n".join([f"+{line}" for line in decoded_content.split('\n')])
                                                file_diffs[file_path] = formatted_diff
                                                continue
                                            except Exception as decode_err:
                                                print(f"Warning: Could not decode content for {file_path}: {str(decode_err)}")
                                    except Exception as api_err:
                                        print(f"Warning: Could not get raw file content for {file_path}: {str(api_err)}")

                                    # Use diff content as fallback
                                    file_diffs[file_path] = diff_content
                            # For deleted files, get the content from the parent commit
                            elif file_diff.get('deleted_file', False):
                                try:
                                    # Get parent commit
                                    parent_commits = project.commits.get(commit.id).parent_ids
                                    if parent_commits:
                                        # Get the file content and handle both string and bytes
                                        try:
                                            file_obj = project.files.get(file_path=old_path, ref=parent_commits[0])
                                            if hasattr(file_obj, 'content'):
                                                # Raw content from API
                                                file_content = file_obj.content
                                            elif hasattr(file_obj, 'decode'):
                                                # Decode if it's bytes
                                                try:
                                                    file_content = file_obj.decode()
                                                except TypeError:
                                                    # If decode fails, try to get content directly
                                                    file_content = file_obj.content if hasattr(file_obj, 'content') else str(file_obj)
                                            else:
                                                # Fallback to string representation
                                                file_content = str(file_obj)

                                            # Format as a proper diff with the entire file as deleted
                                            formatted_diff = f"diff --git a/{old_path} b/{file_path}\n--- a/{old_path}\n+++ /dev/null\n"
                                            formatted_diff += "\n".join([f"-{line}" for line in file_content.split('\n')])
                                            file_diffs[file_path] = formatted_diff
                                        except Exception as file_err:
                                            # Try to get the raw file content directly from the API
                                            try:
                                                import base64
                                                raw_file = project.repository_files.get(file_path=old_path, ref=parent_commits[0])
                                                if raw_file and hasattr(raw_file, 'content'):
                                                    # Decode base64 content if available
                                                    try:
                                                        decoded_content = base64.b64decode(raw_file.content).decode('utf-8', errors='replace')
                                                        formatted_diff = f"diff --git a/{old_path} b/{file_path}\n--- a/{old_path}\n+++ /dev/null\n"
                                                        formatted_diff += "\n".join([f"-{line}" for line in decoded_content.split('\n')])
                                                        file_diffs[file_path] = formatted_diff
                                                    except Exception as decode_err:
                                                        print(f"Warning: Could not decode content for deleted file {old_path}: {str(decode_err)}")
                                                        file_diffs[file_path] = diff_content
                                                else:
                                                    file_diffs[file_path] = diff_content
                                            except Exception as api_err:
                                                print(f"Warning: Could not get raw file content for deleted file {old_path}: {str(api_err)}")
                                                file_diffs[file_path] = diff_content
                                    else:
                                        file_diffs[file_path] = diff_content
                                except Exception as e:
                                    print(f"Warning: Could not get content for deleted file {old_path}: {str(e)}")
                                    file_diffs[file_path] = diff_content
                            # For modified files, use the diff content
                            else:
                                # Check if diff_content is empty or minimal
                                if not diff_content or len(diff_content.strip()) < 10:
                                    # Try to get the full file content for better context
                                    try:
                                        # Get the file content and handle both string and bytes
                                        file_obj = project.files.get(file_path=file_path, ref=commit.id)
                                        if hasattr(file_obj, 'content'):
                                            # Raw content from API
                                            file_content = file_obj.content
                                        elif hasattr(file_obj, 'decode'):
                                            # Decode if it's bytes
                                            try:
                                                file_content = file_obj.decode()
                                            except TypeError:
                                                # If decode fails, try to get content directly
                                                file_content = file_obj.content if hasattr(file_obj, 'content') else str(file_obj)
                                        else:
                                            # Fallback to string representation
                                            file_content = str(file_obj)

                                        # Format as a proper diff with the entire file
                                        formatted_diff = f"diff --git a/{old_path} b/{file_path}\n--- a/{old_path}\n+++ b/{file_path}\n"
                                        formatted_diff += "\n".join([f"+{line}" for line in file_content.split('\n')])
                                        file_diffs[file_path] = formatted_diff
                                    except Exception as e:
                                        print(f"Warning: Could not get content for modified file {file_path}: {str(e)}")
                                        # Try to get the raw file content directly from the API
                                        try:
                                            import base64
                                            raw_file = project.repository_files.get(file_path=file_path, ref=commit.id)
                                            if raw_file and hasattr(raw_file, 'content'):
                                                # Decode base64 content if available
                                                try:
                                                    decoded_content = base64.b64decode(raw_file.content).decode('utf-8', errors='replace')
                                                    formatted_diff = f"diff --git a/{old_path} b/{file_path}\n--- a/{old_path}\n+++ b/{file_path}\n"
                                                    formatted_diff += "\n".join([f"+{line}" for line in decoded_content.split('\n')])
                                                    file_diffs[file_path] = formatted_diff
                                                except Exception as decode_err:
                                                    print(f"Warning: Could not decode content for {file_path}: {str(decode_err)}")
                                                    # Enhance the diff format with what we have
                                                    formatted_diff = f"diff --git a/{old_path} b/{file_path}\n--- a/{old_path}\n+++ b/{file_path}\n{diff_content}"
                                                    file_diffs[file_path] = formatted_diff
                                            else:
                                                # Enhance the diff format with what we have
                                                formatted_diff = f"diff --git a/{old_path} b/{file_path}\n--- a/{old_path}\n+++ b/{file_path}\n{diff_content}"
                                                file_diffs[file_path] = formatted_diff
                                        except Exception as api_err:
                                            print(f"Warning: Could not get raw file content for {file_path}: {str(api_err)}")
                                            # Enhance the diff format with what we have
                                            formatted_diff = f"diff --git a/{old_path} b/{file_path}\n--- a/{old_path}\n+++ b/{file_path}\n{diff_content}"
                                            file_diffs[file_path] = formatted_diff
                                else:
                                    # Enhance the diff format
                                    formatted_diff = f"diff --git a/{old_path} b/{file_path}\n--- a/{old_path}\n+++ b/{file_path}\n{diff_content}"
                                    file_diffs[file_path] = formatted_diff
                        except Exception as e:
                            print(f"Warning: Error processing diff for {file_path}: {str(e)}")
                            file_diffs[file_path] = diff_content

                    # Skip if no valid diffs
                    if not file_diffs:
                        continue

                    # Create CommitInfo object with enhanced diff content
                    commit_info = CommitInfo(
                        hash=commit.id,
                        author=commit.author_name,
                        date=datetime.strptime(commit.created_at, "%Y-%m-%dT%H:%M:%S.%f%z"),
                        message=commit.message,
                        files=list(file_diffs.keys()),
                        diff="\n\n".join(file_diffs.values()),
                        added_lines=sum(diff.count('\n+') for diff in file_diffs.values()),
                        deleted_lines=sum(diff.count('\n-') for diff in file_diffs.values()),
                        effective_lines=sum(diff.count('\n+') - diff.count('\n-') for diff in file_diffs.values())
                    )
                    commits.append(commit_info)

                    # Store file diffs for this commit
                    commit_file_diffs[commit.id] = file_diffs

            # Calculate code stats
            code_stats = {
                "total_added_lines": sum(commit.added_lines for commit in commits),
                "total_deleted_lines": sum(commit.deleted_lines for commit in commits),
                "total_effective_lines": sum(commit.effective_lines for commit in commits),
                "total_files": len(set(file for commit in commits for file in commit.files))
            }

            return commits, commit_file_diffs, code_stats

        except Exception as e:
            error_msg = f"Failed to retrieve GitLab commits: {str(e)}"
            print(error_msg)
            return [], {}, {}

    else:
        error_msg = f"Unsupported platform: {platform}. Use 'github' or 'gitlab'."
        print(error_msg)
        return [], {}, {}


async def evaluate_repository_code(
    repo_path: str,
    start_date: str,
    end_date: str,
    include_extensions: Optional[List[str]] = None,
    exclude_extensions: Optional[List[str]] = None,
    model_name: str = "gpt-3.5",
    output_dir: Optional[str] = None,
    email_addresses: Optional[List[str]] = None,
    platform: str = "local",
    gitlab_url: Optional[str] = None,
):
    """Evaluate all commits in a repository within a time period for all committers.

    Args:
        repo_path: Repository path or name (e.g. owner/repo for remote repositories)
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        include_extensions: List of file extensions to include
        exclude_extensions: List of file extensions to exclude
        model_name: Name of the model to use for evaluation
        output_dir: Directory to save reports
        email_addresses: List of email addresses to send the report to
        platform: Platform to use (github, gitlab, or local)
        gitlab_url: GitLab URL (for GitLab platform only)

    Returns:
        Dict[str, str]: Dictionary mapping author names to their report paths
    """
    # Generate default output directory if not provided
    if not output_dir:
        date_slug = datetime.now().strftime("%Y%m%d")
        output_dir = f"codedog_repo_eval_{date_slug}"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get model
    model = load_model_by_name(model_name)

    print(f"Evaluating repository {repo_path} commits from {start_date} to {end_date}...")

    # Get all commits grouped by author
    if platform.lower() == "local":
        # For local repositories, we need to get all authors first
        # This is a simplified implementation - in a real scenario, you'd need to implement
        # a function to get all authors from a local git repository
        print("Local repository evaluation not implemented yet. Please use github or gitlab platform.")
        return {}
    else:
        # Use remote repository (GitHub or GitLab)
        author_commits = get_all_remote_commits(
            platform,
            repo_path,
            start_date,
            end_date,
            include_extensions,
            exclude_extensions,
            gitlab_url
        )

    if not author_commits:
        print(f"No commits found in repository {repo_path} for the specified time period")
        return {}

    print(f"Found commits from {len(author_commits)} authors in the repository")

    # Initialize evaluator
    evaluator = DiffEvaluator(model)

    # Dictionary to store report paths for each author
    author_reports = {}

    # Summary of all authors
    summary_report = f"# Repository Evaluation Summary\n\n"
    summary_report += f"## Repository: {repo_path}\n"
    summary_report += f"## Period: {start_date} to {end_date}\n\n"
    summary_report += f"## Authors\n\n"

    # Process each author's commits
    for author, (commits, commit_file_diffs, code_stats) in author_commits.items():
        if not commits:
            continue

        print(f"\nEvaluating {len(commits)} commits by {author}...")

        # Generate output file name for this author
        author_slug = author.replace("@", "_at_").replace(" ", "_").replace("/", "_").replace("<", "").replace(">", "")
        output_file = os.path.join(output_dir, f"codedog_eval_{author_slug}.md")

        # Timing and statistics
        start_time = time.time()

        try:
            with get_openai_callback() as cb:
                # Perform evaluation
                print(f"Evaluating code commits for {author}...")
                evaluation_results = await evaluator.evaluate_commits(commits, commit_file_diffs)

                # Generate Markdown report
                report = generate_evaluation_markdown(evaluation_results)

                # Calculate cost and tokens
                total_cost = cb.total_cost
                total_tokens = cb.total_tokens

            # Add evaluation statistics
            elapsed_time = time.time() - start_time
            telemetry_info = (
                f"\n## Evaluation Statistics\n\n"
                f"- **Evaluation Model**: {model_name}\n"
                f"- **Evaluation Time**: {elapsed_time:.2f} seconds\n"
                f"- **Tokens Used**: {total_tokens}\n"
                f"- **Cost**: ${total_cost:.4f}\n"
                f"\n## Code Statistics\n\n"
                f"- **Total Files Modified**: {code_stats.get('total_files', 0)}\n"
                f"- **Lines Added**: {code_stats.get('total_added_lines', 0)}\n"
                f"- **Lines Deleted**: {code_stats.get('total_deleted_lines', 0)}\n"
                f"- **Effective Lines**: {code_stats.get('total_effective_lines', 0)}\n"
            )

            report += telemetry_info

            # Save report immediately after evaluation is complete
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(report)

            # Add to author reports dictionary
            author_reports[author] = output_file

            # Print completion message with clear indication that the file is ready
            print(f"\n✅ Evaluation for {author} completed and saved to {output_file}")
            print(f"   - Files: {code_stats.get('total_files', 0)}")
            print(f"   - Lines: +{code_stats.get('total_added_lines', 0)}/-{code_stats.get('total_deleted_lines', 0)}")
            print(f"   - Time: {elapsed_time:.2f} seconds")
            print(f"   - Cost: ${total_cost:.4f}")

            # Add to summary report
            summary_report += f"### {author}\n\n"
            summary_report += f"- **Commits**: {len(commits)}\n"
            summary_report += f"- **Files Modified**: {code_stats.get('total_files', 0)}\n"
            summary_report += f"- **Lines Added**: {code_stats.get('total_added_lines', 0)}\n"
            summary_report += f"- **Lines Deleted**: {code_stats.get('total_deleted_lines', 0)}\n"
            summary_report += f"- **Effective Lines**: {code_stats.get('total_effective_lines', 0)}\n"
            summary_report += f"- **Report**: [{os.path.basename(output_file)}]({os.path.basename(output_file)})\n\n"

            # Update the summary file after each committer is evaluated
            summary_file = os.path.join(output_dir, "summary.md")
            with open(summary_file, "w", encoding="utf-8") as f:
                f.write(summary_report)
            logger.info(f"Updated summary report with {author}'s evaluation")

        except Exception as e:
            # Log the error but continue with other authors
            error_msg = f"Error evaluating {author}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            print(f"\n❌ Error evaluating {author}: {str(e)}")

            # Create an error report for this author
            error_report = f"# Evaluation Error for {author}\n\n"
            error_report += f"## Error Details\n\n"
            error_report += f"```\n{str(e)}\n```\n\n"
            error_report += f"## Commit Statistics\n\n"
            error_report += f"- **Commits**: {len(commits)}\n"
            error_report += f"- **Files Modified**: {code_stats.get('total_files', 0)}\n"
            error_report += f"- **Lines Added**: {code_stats.get('total_added_lines', 0)}\n"
            error_report += f"- **Lines Deleted**: {code_stats.get('total_deleted_lines', 0)}\n"

            # Save the error report
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(error_report)

            # Add to author reports dictionary
            author_reports[author] = output_file

            # Add error entry to summary report
            summary_report += f"### {author} ❌\n\n"
            summary_report += f"- **Status**: Error during evaluation\n"
            summary_report += f"- **Commits**: {len(commits)}\n"
            summary_report += f"- **Files Modified**: {code_stats.get('total_files', 0)}\n"
            summary_report += f"- **Report**: [{os.path.basename(output_file)}]({os.path.basename(output_file)})\n\n"

            # Update the summary file after each committer is evaluated (even if there's an error)
            summary_file = os.path.join(output_dir, "summary.md")
            with open(summary_file, "w", encoding="utf-8") as f:
                f.write(summary_report)
            logger.info(f"Updated summary report with error for {author}")

    # Final summary report is already saved incrementally, just print a message
    summary_file = os.path.join(output_dir, "summary.md")
    print(f"\nFinal summary report saved to {summary_file}")

    # Send email report if addresses provided
    if email_addresses:
        subject = f"[CodeDog] Repository Evaluation Report for {repo_path} ({start_date} to {end_date})"

        sent = send_report_email(
            to_emails=email_addresses,
            subject=subject,
            markdown_content=summary_report,
        )

        if sent:
            print(f"Summary report sent to {', '.join(email_addresses)}")
        else:
            print("Failed to send email notification")

    return author_reports

async def evaluate_developer_code(
    author: str,
    start_date: str,
    end_date: str,
    repo_path: Optional[str] = None,
    include_extensions: Optional[List[str]] = None,
    exclude_extensions: Optional[List[str]] = None,
    model_name: str = "gpt-3.5",
    output_file: Optional[str] = None,
    email_addresses: Optional[List[str]] = None,
    platform: str = "local",
    gitlab_url: Optional[str] = None,
):
    """Evaluate a developer's code commits in a time period."""
    # Generate default output file name if not provided
    if not output_file:
        author_slug = author.replace("@", "_at_").replace(" ", "_").replace("/", "_")
        date_slug = datetime.now().strftime("%Y%m%d")
        output_file = f"codedog_eval_{author_slug}_{date_slug}.md"

    # Get model
    model = load_model_by_name(model_name)

    print(f"Evaluating {author}'s code commits from {start_date} to {end_date}...")

    # Get commits and diffs based on platform
    if platform.lower() == "local":
        # Use local git repository
        commits, commit_file_diffs, code_stats = get_file_diffs_by_timeframe(
            author,
            start_date,
            end_date,
            repo_path,
            include_extensions,
            exclude_extensions
        )
    else:
        # Use remote repository (GitHub or GitLab)
        if not repo_path:
            print("Repository path/name is required for remote platforms")
            return

        commits, commit_file_diffs, code_stats = get_remote_commits(
            platform,
            repo_path,
            author,
            start_date,
            end_date,
            include_extensions,
            exclude_extensions,
            gitlab_url
        )

    if not commits:
        print(f"No commits found for {author} in the specified time period")
        return

    print(f"Found {len(commits)} commits with {sum(len(diffs) for diffs in commit_file_diffs.values())} modified files")

    # Initialize evaluator
    evaluator = DiffEvaluator(model)

    # Timing and statistics
    start_time = time.time()

    with get_openai_callback() as cb:
        # Perform evaluation
        print("Evaluating code commits...")
        evaluation_results = await evaluator.evaluate_commits(commits, commit_file_diffs)

        # Generate Markdown report
        report = generate_evaluation_markdown(evaluation_results)

        # Calculate cost and tokens
        total_cost = cb.total_cost
        total_tokens = cb.total_tokens

    # Add evaluation statistics
    elapsed_time = time.time() - start_time
    telemetry_info = (
        f"\n## Evaluation Statistics\n\n"
        f"- **Evaluation Model**: {model_name}\n"
        f"- **Evaluation Time**: {elapsed_time:.2f} seconds\n"
        f"- **Tokens Used**: {total_tokens}\n"
        f"- **Cost**: ${total_cost:.4f}\n"
        f"\n## Code Statistics\n\n"
        f"- **Total Files Modified**: {code_stats.get('total_files', 0)}\n"
        f"- **Lines Added**: {code_stats.get('total_added_lines', 0)}\n"
        f"- **Lines Deleted**: {code_stats.get('total_deleted_lines', 0)}\n"
        f"- **Effective Lines**: {code_stats.get('total_effective_lines', 0)}\n"
    )

    report += telemetry_info

    # Save report
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report saved to {output_file}")

    # Send email report if addresses provided
    if email_addresses:
        subject = f"[CodeDog] Code Evaluation Report for {author} ({start_date} to {end_date})"

        sent = send_report_email(
            to_emails=email_addresses,
            subject=subject,
            markdown_content=report,
        )

        if sent:
            print(f"Report sent to {', '.join(email_addresses)}")
        else:
            print("Failed to send email notification")

    return report


def generate_full_report(repository_name, pull_request_number, email_addresses=None, platform="github", gitlab_url=None):
    """Generate a full report including PR summary and code review.

    Args:
        repository_name (str): Repository path (e.g. owner/repo)
        pull_request_number (int): Pull request number to review
        email_addresses (list, optional): List of email addresses to send the report to
        platform (str, optional): Platform to use (github or gitlab). Defaults to "github".
        gitlab_url (str, optional): GitLab URL. Defaults to https://gitlab.com or GITLAB_URL env var.
    """
    start_time = time.time()

    # Initialize client and retriever based on platform
    if platform.lower() == "github":
        # Initialize GitHub client and retriever
        github_client = Github()  # Will automatically load GITHUB_TOKEN from environment
        print(f"Analyzing GitHub repository {repository_name} PR #{pull_request_number}")

        try:
            retriever = GithubRetriever(github_client, repository_name, pull_request_number)
            print(f"Successfully retrieved PR: {retriever.pull_request.title}")
        except Exception as e:
            error_msg = f"Failed to retrieve GitHub PR: {str(e)}"
            print(error_msg)
            return error_msg

    elif platform.lower() == "gitlab":
        # Initialize GitLab client and retriever
        gitlab_token = os.environ.get("GITLAB_TOKEN", "")
        if not gitlab_token:
            error_msg = "GITLAB_TOKEN environment variable is not set"
            print(error_msg)
            return error_msg

        # Use provided GitLab URL or fall back to environment variable or default
        gitlab_url = gitlab_url or os.environ.get("GITLAB_URL", "https://gitlab.com")

        gitlab_client = Gitlab(url=gitlab_url, private_token=gitlab_token)
        print(f"Analyzing GitLab repository {repository_name} MR #{pull_request_number}")

        try:
            retriever = GitlabRetriever(gitlab_client, repository_name, pull_request_number)
            print(f"Successfully retrieved MR: {retriever.pull_request.title}")
        except Exception as e:
            error_msg = f"Failed to retrieve GitLab MR: {str(e)}"
            print(error_msg)
            return error_msg

    else:
        error_msg = f"Unsupported platform: {platform}. Use 'github' or 'gitlab'."
        print(error_msg)
        return error_msg

    # Load models based on environment variables
    code_summary_model = os.environ.get("CODE_SUMMARY_MODEL", "gpt-3.5")
    pr_summary_model = os.environ.get("PR_SUMMARY_MODEL", "gpt-4")
    code_review_model = os.environ.get("CODE_REVIEW_MODEL", "gpt-3.5")

    # Initialize chains with specified models
    summary_chain = PRSummaryChain.from_llm(
        code_summary_llm=load_model_by_name(code_summary_model),
        pr_summary_llm=load_model_by_name(pr_summary_model),
        verbose=True
    )

    review_chain = CodeReviewChain.from_llm(
        llm=load_model_by_name(code_review_model),
        verbose=True
    )

    with get_openai_callback() as cb:
        # Get PR summary
        print(f"Generating PR summary using {pr_summary_model}...")
        pr_summary_result = asyncio.run(pr_summary(retriever, summary_chain))
        pr_summary_cost = cb.total_cost
        print(f"PR summary complete, cost: ${pr_summary_cost:.4f}")

        # Get code review
        print(f"Generating code review using {code_review_model}...")
        try:
            code_review_result = asyncio.run(code_review(retriever, review_chain))
            code_review_cost = cb.total_cost - pr_summary_cost
            print(f"Code review complete, cost: ${code_review_cost:.4f}")
        except Exception as e:
            print(f"Code review generation failed: {str(e)}")
            print(traceback.format_exc())
            # Use empty code review
            code_review_result = {"code_reviews": []}

        # Create report
        total_cost = cb.total_cost
        total_time = time.time() - start_time

        reporter = PullRequestReporter(
            pr_summary=pr_summary_result["pr_summary"],
            code_summaries=pr_summary_result["code_summaries"],
            pull_request=retriever.pull_request,
            code_reviews=code_review_result.get("code_reviews", []),
            telemetry={
                "start_time": start_time,
                "time_usage": total_time,
                "cost": total_cost,
                "tokens": cb.total_tokens,
            },
        )

        report = reporter.report()

        # Save report to file
        report_file = f"codedog_pr_{pull_request_number}.md"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Report saved to {report_file}")

        # Send email notification if email addresses provided
        if email_addresses:
            subject = f"[CodeDog] Code Review for {repository_name} PR #{pull_request_number}: {retriever.pull_request.title}"
            sent = send_report_email(
                to_emails=email_addresses,
                subject=subject,
                markdown_content=report,
            )
            if sent:
                print(f"Report sent to {', '.join(email_addresses)}")
            else:
                print("Failed to send email notification")

        return report


async def review_commit(
    commit_hash: str,
    repo_path: Optional[str] = None,
    include_extensions: Optional[List[str]] = None,
    exclude_extensions: Optional[List[str]] = None,
    model_name: str = "gpt-3.5",
    output_file: Optional[str] = None,
    email_addresses: Optional[List[str]] = None,
    platform: str = "local",
    gitlab_url: Optional[str] = None,
):
    """Review a specific commit.

    Args:
        commit_hash: The hash of the commit to review
        repo_path: Git repository path or name (e.g. owner/repo for remote repositories)
        include_extensions: List of file extensions to include (e.g. ['.py', '.js'])
        exclude_extensions: List of file extensions to exclude (e.g. ['.md', '.txt'])
        model_name: Name of the model to use for review
        output_file: Path to save the report to
        email_addresses: List of email addresses to send the report to
        platform: Platform to use (github, gitlab, or local)
        gitlab_url: GitLab URL (for GitLab platform only)
    """
    logger.info(f"Starting commit review for {commit_hash}")
    logger.info(f"Parameters: repo_path={repo_path}, platform={platform}, model={model_name}")
    logger.info(f"Include extensions: {include_extensions}, Exclude extensions: {exclude_extensions}")

    # Generate default output file name if not provided
    if not output_file:
        date_slug = datetime.now().strftime("%Y%m%d")
        output_file = f"codedog_commit_{commit_hash[:8]}_{date_slug}.md"
        logger.info(f"Generated output file name: {output_file}")
    else:
        logger.info(f"Using provided output file: {output_file}")

    # Get model
    logger.info(f"Loading model: {model_name}")
    model = load_model_by_name(model_name)
    logger.info(f"Model loaded: {model.__class__.__name__}")

    print(f"Reviewing commit {commit_hash}...")

    # Get commit diff based on platform
    commit_diff = {}

    if platform.lower() == "local":
        # Use local git repository
        logger.info(f"Using local git repository: {repo_path or 'current directory'}")
        try:
            logger.info(f"Getting commit diff for {commit_hash}")
            commit_diff = get_commit_diff(commit_hash, repo_path, include_extensions, exclude_extensions)
            logger.info(f"Successfully retrieved commit diff with {len(commit_diff)} files")
        except Exception as e:
            error_msg = f"Error getting commit diff: {str(e)}"
            logger.error(error_msg, exc_info=True)
            print(error_msg)
            return
    elif platform.lower() in ["github", "gitlab"]:
        # Use remote repository
        logger.info(f"Using remote {platform} repository: {repo_path}")
        if not repo_path or "/" not in repo_path:
            error_msg = f"Error: Repository name must be in the format 'owner/repo' for {platform} platform"
            logger.error(error_msg)
            print(error_msg)
            return

        logger.info(f"Getting remote commit diff for {commit_hash} from {platform}")
        commit_diff = get_remote_commit_diff(
            platform=platform,
            repository_name=repo_path,
            commit_hash=commit_hash,
            include_extensions=include_extensions,
            exclude_extensions=exclude_extensions,
            gitlab_url=gitlab_url,
        )
        logger.info(f"Retrieved remote commit diff with {len(commit_diff)} files")
    else:
        error_msg = f"Error: Unsupported platform '{platform}'. Use 'local', 'github', or 'gitlab'."
        logger.error(error_msg)
        print(error_msg)
        return

    if not commit_diff:
        logger.warning(f"No changes found in commit {commit_hash}")
        print(f"No changes found in commit {commit_hash}")
        return

    # Log detailed information about the files
    logger.info(f"Found {len(commit_diff)} modified files:")
    for file_path, diff_info in commit_diff.items():
        logger.info(f"  - {file_path} (status: {diff_info.get('status', 'unknown')}, " +
                   f"additions: {diff_info.get('additions', 0)}, " +
                   f"deletions: {diff_info.get('deletions', 0)})")
        # Log the size of the diff content
        diff_content = diff_info.get('diff', '')
        logger.debug(f"    Diff content size: {len(diff_content)} characters, " +
                    f"~{len(diff_content.split())} words")

    print(f"Found {len(commit_diff)} modified files")

    # Initialize evaluator
    logger.info("Initializing DiffEvaluator")
    evaluator = DiffEvaluator(model)
    logger.info(f"DiffEvaluator initialized with model: {model.__class__.__name__}")

    # Timing and statistics
    start_time = time.time()
    logger.info(f"Starting evaluation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    with get_openai_callback() as cb:
        # Perform review
        print("Reviewing code changes...")
        logger.info(f"Starting commit evaluation for {commit_hash}")
        review_results = await evaluator.evaluate_commit(commit_hash, commit_diff)
        logger.info(f"Commit evaluation completed, got results for {len(review_results.get('files', []))} files")

        # Log evaluation results summary
        logger.info(f"Statistics: {review_results.get('statistics', {})}")

        # Generate Markdown report
        logger.info("Generating Markdown report")
        report = f"# Commit Review Report\n\n"
        report += f"## Commit: {commit_hash}\n\n"
        report += f"## Summary\n\n{review_results.get('summary', 'No summary available.')}\n\n"
        report += f"## Statistics\n\n"

        # Get statistics
        total_files = review_results.get('statistics', {}).get('total_files', 0)
        total_additions = review_results.get('statistics', {}).get('total_additions', 0)
        total_deletions = review_results.get('statistics', {}).get('total_deletions', 0)

        report += f"- Total files: {total_files}\n"
        report += f"- Total additions: {total_additions}\n"
        report += f"- Total deletions: {total_deletions}\n\n"
        report += f"## Files\n\n"

        # Log detailed file statistics
        logger.info(f"Report statistics: {total_files} files, {total_additions} additions, {total_deletions} deletions")

        for file in review_results.get('files', []):
            file_path = file.get('path', 'Unknown file')
            file_status = file.get('status', 'Unknown')
            file_additions = file.get('additions', 0)
            file_deletions = file.get('deletions', 0)
            overall_score = file.get('overall_score', 'N/A')

            logger.info(f"File report: {file_path} (status: {file_status}, score: {overall_score})")

            report += f"### {file_path}\n\n"
            report += f"- Status: {file_status}\n"
            report += f"- Additions: {file_additions}\n"
            report += f"- Deletions: {file_deletions}\n"
            report += f"- Overall Score: {overall_score}\n\n"
            report += f"#### Scores\n\n"

            # Get all scores
            readability = file.get('readability', 'N/A')
            efficiency = file.get('efficiency', 'N/A')
            security = file.get('security', 'N/A')
            structure = file.get('structure', 'N/A')
            error_handling = file.get('error_handling', 'N/A')
            documentation = file.get('documentation', 'N/A')
            code_style = file.get('code_style', 'N/A')

            # Log detailed scores
            logger.debug(f"File scores: {file_path} - " +
                        f"readability: {readability}, " +
                        f"efficiency: {efficiency}, " +
                        f"security: {security}, " +
                        f"structure: {structure}, " +
                        f"error_handling: {error_handling}, " +
                        f"documentation: {documentation}, " +
                        f"code_style: {code_style}")

            report += f"- Readability: {readability}\n"
            report += f"- Efficiency: {efficiency}\n"
            report += f"- Security: {security}\n"
            report += f"- Structure: {structure}\n"
            report += f"- Error Handling: {error_handling}\n"
            report += f"- Documentation: {documentation}\n"
            report += f"- Code Style: {code_style}\n\n"

            comments = file.get('comments', 'No comments.')
            report += f"#### Comments\n\n{comments}\n\n"
            report += f"---\n\n"

        # Calculate cost and tokens
        total_cost = cb.total_cost
        total_tokens = cb.total_tokens
        logger.info(f"API usage: {total_tokens} tokens, ${total_cost:.4f}")

    # Add review statistics
    elapsed_time = time.time() - start_time
    logger.info(f"Review completed in {elapsed_time:.2f} seconds")

    # Add whole commit evaluation section if available
    if "whole_commit_evaluation" in review_results:
        whole_eval = review_results["whole_commit_evaluation"]
        whole_eval_section = f"\n## Whole Commit Evaluation\n\n"
        whole_eval_section += f"### Scores\n\n"
        whole_eval_section += f"| Dimension | Score |\n"
        whole_eval_section += f"|-----------|-------|\n"
        whole_eval_section += f"| Readability | {whole_eval.get('readability', 'N/A')}/10 |\n"
        whole_eval_section += f"| Efficiency | {whole_eval.get('efficiency', 'N/A')}/10 |\n"
        whole_eval_section += f"| Security | {whole_eval.get('security', 'N/A')}/10 |\n"
        whole_eval_section += f"| Structure | {whole_eval.get('structure', 'N/A')}/10 |\n"
        whole_eval_section += f"| Error Handling | {whole_eval.get('error_handling', 'N/A')}/10 |\n"
        whole_eval_section += f"| Documentation | {whole_eval.get('documentation', 'N/A')}/10 |\n"
        whole_eval_section += f"| Code Style | {whole_eval.get('code_style', 'N/A')}/10 |\n"
        whole_eval_section += f"| **Overall Score** | **{whole_eval.get('overall_score', 'N/A')}/10** |\n\n"

        # Add analysis from whole commit evaluation
        whole_eval_section += f"### Analysis\n\n{whole_eval.get('comments', 'No comments available.')}\n\n"

        # Insert the whole commit evaluation section after the summary
        report = report.replace("## Files\n\n", whole_eval_section + "## Files\n\n")

    # Add estimated working hours if available
    estimated_hours_info = ""
    if "estimated_hours" in review_results:
        estimated_hours_info = (
            f"- **Estimated Working Hours**: {review_results['estimated_hours']} hours "
            f"(for an experienced programmer with 5-10+ years of experience)\n"
        )

    telemetry_info = (
        f"\n## Review Statistics\n\n"
        f"- **Review Model**: {model_name}\n"
        f"- **Review Time**: {elapsed_time:.2f} seconds\n"
        f"- **Tokens Used**: {total_tokens}\n"
        f"- **Cost**: ${total_cost:.4f}\n"
        f"\n## Code Statistics\n\n"
        f"- **Total Files Modified**: {len(commit_diff)}\n"
        f"- **Lines Added**: {sum(diff.get('additions', 0) for diff in commit_diff.values())}\n"
        f"- **Lines Deleted**: {sum(diff.get('deletions', 0) for diff in commit_diff.values())}\n"
        f"{estimated_hours_info}"
    )

    report += telemetry_info

    # Save report
    logger.info(f"Saving report to {output_file}")
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"Report successfully saved to {output_file}")
        print(f"Report saved to {output_file}")
    except Exception as e:
        error_msg = f"Error saving report to {output_file}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        print(error_msg)

    # Send email report if addresses provided
    if email_addresses:
        logger.info(f"Sending report to {email_addresses}")
        print(f"Sending report to {', '.join(email_addresses)}...")
        subject = f"[CodeDog] Code Review for Commit {commit_hash[:8]}"

        try:
            sent = send_report_email(
                to_emails=email_addresses,
                subject=subject,
                markdown_content=report,
            )

            if sent:
                logger.info(f"Report successfully sent to {email_addresses}")
                print(f"Report sent to {', '.join(email_addresses)}")
            else:
                logger.error("Failed to send email notification")
                print("Failed to send email notification")
        except Exception as e:
            error_msg = f"Error sending email: {str(e)}"
            logger.error(error_msg, exc_info=True)
            print(error_msg)

    logger.info("Commit review completed successfully")
    return report


def main():
    """Main function to parse arguments and run the appropriate command."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("codedog.log"),
            logging.StreamHandler()
        ]
    )
    logger.info("Starting CodeDog")

    # Parse arguments
    args = parse_args()
    logger.info(f"Command: {args.command}")
    logger.debug(f"Arguments: {args}")

    if args.command == "pr":
        # Review a GitHub or GitLab pull request
        email_addresses = parse_emails(args.email or os.environ.get("NOTIFICATION_EMAILS", ""))
        report = generate_full_report(
            repository_name=args.repository,
            pull_request_number=args.pr_number,
            email_addresses=email_addresses,
            platform=args.platform,
            gitlab_url=args.gitlab_url
        )

        print("\n===================== Review Report =====================\n")
        print(report)
        print("\n===================== Report End =====================\n")

    elif args.command == "setup-hooks":
        # Set up git hooks for commit-triggered reviews
        repo_path = args.repo or os.getcwd()
        success = install_git_hooks(repo_path)
        if success:
            print("Git hooks successfully installed.")
            print("CodeDog will now automatically review new commits.")

            # Check if notification emails are configured
            emails = os.environ.get("NOTIFICATION_EMAILS", "")
            if emails:
                print(f"Notification emails configured: {emails}")
            else:
                print("No notification emails configured. Add NOTIFICATION_EMAILS to your .env file to receive email reports.")
        else:
            print("Failed to install git hooks.")

    elif args.command == "eval":
        # Evaluate developer's code commits
        # Process date parameters
        today = datetime.now().strftime("%Y-%m-%d")
        week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        start_date = args.start_date or week_ago
        end_date = args.end_date or today

        # Process file extension parameters
        include_extensions = None
        if args.include:
            include_extensions = parse_extensions(args.include)
        elif os.environ.get("DEV_EVAL_DEFAULT_INCLUDE"):
            include_extensions = parse_extensions(os.environ.get("DEV_EVAL_DEFAULT_INCLUDE"))

        exclude_extensions = None
        if args.exclude:
            exclude_extensions = parse_extensions(args.exclude)
        elif os.environ.get("DEV_EVAL_DEFAULT_EXCLUDE"):
            exclude_extensions = parse_extensions(os.environ.get("DEV_EVAL_DEFAULT_EXCLUDE"))

        # Get model
        model_name = args.model or os.environ.get("CODE_REVIEW_MODEL", "gpt-3.5")

        # Get email addresses
        email_addresses = parse_emails(args.email or os.environ.get("NOTIFICATION_EMAILS", ""))

        # Run evaluation
        report = asyncio.run(evaluate_developer_code(
            author=args.author,
            start_date=start_date,
            end_date=end_date,
            repo_path=args.repo,
            include_extensions=include_extensions,
            exclude_extensions=exclude_extensions,
            model_name=model_name,
            output_file=args.output,
            email_addresses=email_addresses,
            platform=args.platform,
            gitlab_url=args.gitlab_url,
        ))

        if report:
            print("\n===================== Evaluation Report =====================\n")
            print("Report generated successfully. See output file for details.")
            print("\n===================== Report End =====================\n")

    elif args.command == "commit":
        logger.info(f"Running commit review for {args.commit_hash}")

        # Process file extension parameters
        include_extensions = None
        if args.include:
            include_extensions = parse_extensions(args.include)
            logger.info(f"Using provided include extensions: {include_extensions}")
        elif os.environ.get("DEV_EVAL_DEFAULT_INCLUDE"):
            include_extensions = parse_extensions(os.environ.get("DEV_EVAL_DEFAULT_INCLUDE"))
            logger.info(f"Using default include extensions from environment: {include_extensions}")

        exclude_extensions = None
        if args.exclude:
            exclude_extensions = parse_extensions(args.exclude)
            logger.info(f"Using provided exclude extensions: {exclude_extensions}")
        elif os.environ.get("DEV_EVAL_DEFAULT_EXCLUDE"):
            exclude_extensions = parse_extensions(os.environ.get("DEV_EVAL_DEFAULT_EXCLUDE"))
            logger.info(f"Using default exclude extensions from environment: {exclude_extensions}")

        # Get model
        model_name = args.model or os.environ.get("CODE_REVIEW_MODEL", "gpt-3.5")
        logger.info(f"Using model: {model_name}")

        # Get email addresses
        email_addresses = parse_emails(args.email or os.environ.get("NOTIFICATION_EMAILS", ""))
        if email_addresses:
            logger.info(f"Will send report to: {email_addresses}")

        # Log platform information
        if args.platform != "local":
            logger.info(f"Using {args.platform} platform with repository: {args.repo}")
            if args.platform == "gitlab" and args.gitlab_url:
                logger.info(f"Using GitLab URL: {args.gitlab_url}")
        else:
            logger.info(f"Using local repository: {args.repo or 'current directory'}")

        # Run commit review
        logger.info("Starting commit review process")
        try:
            report = asyncio.run(review_commit(
                commit_hash=args.commit_hash,
                repo_path=args.repo,
                include_extensions=include_extensions,
                exclude_extensions=exclude_extensions,
                model_name=model_name,
                output_file=args.output,
                email_addresses=email_addresses,
                platform=args.platform,
                gitlab_url=args.gitlab_url,
            ))

            logger.info("Commit review completed successfully")

            if report:
                logger.info("Report generated successfully")
                print("\n===================== Commit Review Report =====================\n")
                print("Report generated successfully. See output file for details.")
                print("\n===================== Report End =====================\n")
        except Exception as e:
            logger.error(f"Error during commit review: {str(e)}", exc_info=True)
            print(f"Error during commit review: {str(e)}")

    elif args.command == "repo-eval":
        logger.info(f"Running repository evaluation for {args.repo}")

        # Set default dates if not provided
        if not args.start_date:
            args.start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
            logger.info(f"Using default start date: {args.start_date}")

        if not args.end_date:
            args.end_date = datetime.now().strftime("%Y-%m-%d")
            logger.info(f"Using default end date: {args.end_date}")

        # Process file extension parameters
        include_extensions = None
        if args.include:
            include_extensions = parse_extensions(args.include)
            logger.info(f"Using provided include extensions: {include_extensions}")
        elif os.environ.get("DEV_EVAL_DEFAULT_INCLUDE"):
            include_extensions = parse_extensions(os.environ.get("DEV_EVAL_DEFAULT_INCLUDE"))
            logger.info(f"Using default include extensions from environment: {include_extensions}")

        exclude_extensions = None
        if args.exclude:
            exclude_extensions = parse_extensions(args.exclude)
            logger.info(f"Using provided exclude extensions: {exclude_extensions}")
        elif os.environ.get("DEV_EVAL_DEFAULT_EXCLUDE"):
            exclude_extensions = parse_extensions(os.environ.get("DEV_EVAL_DEFAULT_EXCLUDE"))
            logger.info(f"Using default exclude extensions from environment: {exclude_extensions}")

        # Get model
        model_name = args.model or os.environ.get("CODE_REVIEW_MODEL", "gpt-3.5")
        logger.info(f"Using model: {model_name}")

        # Get email addresses
        email_addresses = parse_emails(args.email or os.environ.get("NOTIFICATION_EMAILS", ""))
        if email_addresses:
            logger.info(f"Will send report to: {email_addresses}")

        # Log platform information
        if args.platform != "local":
            logger.info(f"Using {args.platform} platform with repository: {args.repo}")
            if args.platform == "gitlab" and args.gitlab_url:
                logger.info(f"Using GitLab URL: {args.gitlab_url}")
        else:
            logger.info(f"Using local repository: {args.repo}")

        # Run repository evaluation
        logger.info("Starting repository evaluation process")
        try:
            author_reports = asyncio.run(evaluate_repository_code(
                repo_path=args.repo,
                start_date=args.start_date,
                end_date=args.end_date,
                include_extensions=include_extensions,
                exclude_extensions=exclude_extensions,
                model_name=model_name,
                output_dir=args.output_dir,
                email_addresses=email_addresses,
                platform=args.platform,
                gitlab_url=args.gitlab_url,
            ))

            logger.info("Repository evaluation completed successfully")

            if author_reports:
                logger.info(f"Generated reports for {len(author_reports)} authors")
                print("\n===================== Repository Evaluation Report =====================\n")
                print(f"Reports generated successfully for {len(author_reports)} authors.")
                print("See output directory for details.")
                print("\n===================== Report End =====================\n")
        except Exception as e:
            logger.error(f"Error during repository evaluation: {str(e)}", exc_info=True)
            print(f"Error during repository evaluation: {str(e)}")

    else:
        # No command specified, show usage
        print("Please specify a command. Use --help for more information.")
        print("Example: python run_codedog.py pr owner/repo 123                      # GitHub PR review")
        print("Example: python run_codedog.py pr owner/repo 123 --platform gitlab    # GitLab MR review")
        print("Example: python run_codedog.py setup-hooks                           # Set up git hooks")
        print("Example: python run_codedog.py eval username --start-date 2023-01-01 --end-date 2023-01-31  # Evaluate code")
        print("Example: python run_codedog.py commit abc123def                      # Review local commit")
        print("Example: python run_codedog.py commit abc123def --repo owner/repo --platform github  # Review GitHub commit")
        print("Example: python run_codedog.py commit abc123def --repo owner/repo --platform gitlab  # Review GitLab commit")
        print("Example: python run_codedog.py repo-eval owner/repo --start-date 2023-01-01 --end-date 2023-01-31 --platform github  # Evaluate all commits in a GitHub repo")
        print("Example: python run_codedog.py repo-eval owner/repo --start-date 2023-01-01 --end-date 2023-01-31 --platform gitlab  # Evaluate all commits in a GitLab repo")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(error_msg)
        print("\nDetailed error information:")
        traceback.print_exc()

        # Log the error if logging is configured
        try:
            logger.error(error_msg, exc_info=True)
            logger.error("Program terminated with error")
        except:
            # If logging is not yet configured, just print
            pass