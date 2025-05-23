# CodeDog Project Updates

## Latest Updates

### 1. Repository Evaluation Command
- Added new `repo-eval` command to evaluate all commits in a repository for all committers
- Implemented incremental report generation - each committer's report is saved immediately after evaluation
- Added summary report with links to individual committer reports
- Added error handling to continue evaluation even if one committer's evaluation fails

### 2. Improved Scoring System
- Enhanced the scoring system to provide more accurate and comprehensive code evaluations
- Added detailed scoring criteria for each dimension
- Implemented weighted scoring for different aspects of code quality

### 3. Evaluation Dimensions
The evaluation now covers the following dimensions:
- Readability: Code clarity and understandability
- Efficiency & Performance: Code execution speed and resource usage
- Security: Code security practices and vulnerability prevention
- Structure & Design: Code organization and architectural design
- Error Handling: Robustness in handling errors and edge cases
- Documentation & Comments: Code documentation quality and completeness
- Code Style: Adherence to coding standards and best practices

### 3. Enhanced Error Handling
- Improved timeout handling for API requests
- Added detailed error logging
- Implemented better error recovery mechanisms

### 4. Performance Optimizations
- Reduced API call latency
- Optimized memory usage
- Improved concurrent request handling

### 5. Documentation Updates
- Added comprehensive API documentation
- Updated user guides
- Improved code examples and tutorials

## Running the Project

### Environment Setup

1. Ensure the .env file is properly configured, especially:
   - Platform tokens (GitHub or GitLab)
   - LLM API keys (OpenAI, DeepSeek, etc.)
   - SMTP server settings (if email notifications are enabled)

2. If using Gmail for email notifications:
   - Enable two-factor authentication for your Google account
   - Generate an app-specific password (https://myaccount.google.com/apppasswords)
   - Use the app password in your .env file

### Running Commands

1. **Evaluate Developer Code**:
   ```bash
   python run_codedog.py eval "developer_name" --start-date YYYY-MM-DD --end-date YYYY-MM-DD
   ```

2. **Review PR/MR**:
   ```bash
   # GitHub PR review
   python run_codedog.py pr "repository_name" PR_number

   # GitLab MR review
   python run_codedog.py pr "repository_name" MR_number --platform gitlab

   # Self-hosted GitLab instance
   python run_codedog.py pr "repository_name" MR_number --platform gitlab --gitlab-url "https://your.gitlab.instance.com"
   ```

3. **Evaluate Repository**:
   ```bash
   # Evaluate all commits in a GitHub repository
   python run_codedog.py repo-eval "repository_name" --start-date YYYY-MM-DD --end-date YYYY-MM-DD --platform github

   # Evaluate all commits in a GitLab repository
   python run_codedog.py repo-eval "repository_name" --start-date YYYY-MM-DD --end-date YYYY-MM-DD --platform gitlab

   # Evaluate with specific model
   python run_codedog.py repo-eval "repository_name" --start-date YYYY-MM-DD --end-date YYYY-MM-DD --platform gitlab --model deepseek
   ```

4. **Set up Git Hooks**:
   ```bash
   python run_codedog.py setup-hooks
   ```

### Important Notes

- For large code diffs, you may encounter context length limits. In such cases, consider using `gpt-4-32k` or other models with larger context windows.
- DeepSeek models have specific message format requirements, please ensure to follow the fixes mentioned above.

## Future Improvements

1. Implement better text chunking and processing for handling large code diffs
2. Develop more specialized scoring criteria for different file types
3. Further improve report presentation with visual charts
4. Deeper integration with CI/CD systems

## TODO
1. Implement better handling of large diffs in GitLab API (currently limited to 20 files per diff)
2. Add support for local repositories in repo-eval command
3. Add more detailed statistics in the summary report