# CodeDog Prompt Testing Tools

这个目录包含两个用于测试代码评审提示(prompts)的工具：

1. `test_prompt.py` - 单个文件或差异的提示测试工具
2. `batch_test_prompts.py` - 从GitLab批量获取差异并测试提示的工具

## 环境设置

确保您已经安装了所需的依赖：

```bash
pip install python-gitlab python-dotenv langchain-openai
```

并创建了一个包含必要环境变量的`.env`文件：

```
# OpenAI API配置
OPENAI_API_KEY=your_openai_api_key

# GitLab配置
GITLAB_URL=https://gitlab.com  # 或您的GitLab实例URL
GITLAB_TOKEN=your_gitlab_token
```

## 单个文件测试工具 (test_prompt.py)

这个工具允许您测试单个文件或差异的代码评审提示。

### 基本用法

1. **评估文件**：
   ```bash
   python test_prompt.py --file example.py
   ```

2. **评估差异文件**：
   ```bash
   python test_prompt.py --diff example.diff
   ```

3. **使用特定模型**：
   ```bash
   python test_prompt.py --diff example.diff --model gpt-4
   ```

4. **使用自定义系统提示**：
   ```bash
   python test_prompt.py --diff example.diff --system-prompt custom_system_prompt.txt
   ```

5. **输出为Markdown格式**：
   ```bash
   python test_prompt.py --diff example.diff --format markdown
   ```

6. **保存输出到文件**：
   ```bash
   python test_prompt.py --diff example.diff --output results.json
   ```

### 命令行选项

```
usage: test_prompt.py [-h] (--file FILE | --diff DIFF) [--model MODEL]
                      [--system-prompt SYSTEM_PROMPT] [--output OUTPUT]
                      [--format {json,markdown}]

Test code review prompts

options:
  -h, --help            show this help message and exit
  --file FILE           Path to the file to evaluate
  --diff DIFF           Path to the diff file to evaluate
  --model MODEL         Model to use for evaluation (default: gpt-3.5-turbo)
  --system-prompt SYSTEM_PROMPT
                        Path to a file containing a custom system prompt
  --output OUTPUT       Path to save the output (default: stdout)
  --format {json,markdown}
                        Output format (default: json)
```

## 批量测试工具 (batch_test_prompts.py)

这个工具允许您从GitLab获取多个差异并批量测试代码评审提示。

### 基本用法

1. **从GitLab获取MR并测试**：
   ```bash
   python batch_test_prompts.py --project your_group/your_project
   ```

2. **指定文件类型**：
   ```bash
   python batch_test_prompts.py --project your_group/your_project --include .py,.js --exclude .md,.txt
   ```

3. **测试多个模型**：
   ```bash
   python batch_test_prompts.py --project your_group/your_project --models gpt-3.5-turbo,gpt-4
   ```

4. **测试多个系统提示**：
   ```bash
   python batch_test_prompts.py --project your_group/your_project --system-prompts prompt1.txt,prompt2.txt
   ```

5. **自定义输出目录和格式**：
   ```bash
   python batch_test_prompts.py --project your_group/your_project --output-dir my_tests --format markdown
   ```

### 命令行选项

```
usage: batch_test_prompts.py [-h] --project PROJECT [--mr-count MR_COUNT]
                             [--max-files MAX_FILES] [--include INCLUDE]
                             [--exclude EXCLUDE]
                             [--state {merged,opened,closed}] [--models MODELS]
                             [--system-prompts SYSTEM_PROMPTS]
                             [--output-dir OUTPUT_DIR]
                             [--format {json,markdown}]

Batch test code review prompts on GitLab MRs

options:
  -h, --help            show this help message and exit
  --project PROJECT     GitLab project ID or path
  --mr-count MR_COUNT   Number of MRs to fetch (default: 5)
  --max-files MAX_FILES
                        Maximum files per MR (default: 3)
  --include INCLUDE     Included file extensions, comma separated, e.g. .py,.js
  --exclude EXCLUDE     Excluded file extensions, comma separated, e.g. .md,.txt
  --state {merged,opened,closed}
                        MR state to fetch (default: merged)
  --models MODELS       Models to test, comma separated (default: gpt-3.5-turbo)
  --system-prompts SYSTEM_PROMPTS
                        Paths to system prompt files, comma separated
  --output-dir OUTPUT_DIR
                        Output directory (default: prompt_tests)
  --format {json,markdown}
                        Output format (default: json)
```

## 输出结果

### 单个文件测试

单个文件测试工具的输出是一个JSON或Markdown文件，包含代码评审结果。

JSON格式示例：
```json
{
  "readability": 8,
  "efficiency": 7,
  "security": 6,
  "structure": 7,
  "error_handling": 5,
  "documentation": 9,
  "code_style": 8,
  "overall_score": 7.1,
  "effective_code_lines": 15,
  "non_effective_code_lines": 5,
  "estimated_hours": 1.5,
  "comments": "详细分析..."
}
```

### 批量测试

批量测试工具的输出是一个目录结构，包含：

1. `diffs/` - 保存从GitLab获取的差异文件
2. `results/` - 保存测试结果
   - 每个模型一个子目录
   - 每个子目录包含每个差异文件的评估结果
   - `summary.json` - 包含该模型所有测试的汇总
3. `comparison_report.md` - 如果测试了多个模型或提示，则生成比较报告

## 自定义系统提示

您可以创建自定义系统提示文件，用于测试不同的提示效果。系统提示文件是一个纯文本文件，包含您想要使用的系统提示。

示例：`custom_system_prompt.txt`

## 提示优化建议

1. **明确角色和目标**：明确定义代码评审员的角色和评审目标。

2. **详细的评估维度**：为每个评估维度提供详细的评估标准。

3. **区分有效和无效代码修改**：明确区分哪些修改是有效的，哪些是无效的。

4. **工作时间估算指南**：提供详细的工作时间估算指南。

5. **结构化输出格式**：明确定义输出格式，确保一致性。

6. **语言特定考虑因素**：为不同的编程语言提供特定的考虑因素。

通过使用这些工具，您可以快速测试和优化代码评审提示，找到最适合您需求的提示。
