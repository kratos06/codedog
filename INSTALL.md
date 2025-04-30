# 安装指南

本文档提供了安装 CodeDog 及其依赖项的详细说明，特别是对于使用 Claude 和 Gemini 模型的用户。

## 安装依赖

### 使用 pip 安装

如果您使用 pip 管理依赖，可以通过以下命令安装所有必要的依赖：

```bash
pip install -r requirements.txt
```

### 使用 Poetry 安装

如果您使用 Poetry 管理依赖，可以通过以下命令安装所有必要的依赖：

```bash
# 安装基本依赖
poetry install

# 安装 Claude 模型支持
poetry add langchain-anthropic

# 安装 Gemini 模型支持
poetry add langchain-google-genai
```

## 模型特定配置

### Claude 模型

要使用 Claude 模型，您需要：

1. 获取 Anthropic API 密钥：访问 [Anthropic 控制台](https://console.anthropic.com/) 注册并获取 API 密钥
2. 在 `.env` 文件中配置以下环境变量：

```
ANTHROPIC_API_KEY="your_anthropic_api_key"
CLAUDE_MODEL="claude-3-sonnet-20240229"  # 可选，默认为 "claude-3-sonnet-20240229"
CLAUDE_TEMPERATURE="0"  # 可选，默认为 0
CLAUDE_MAX_TOKENS="4096"  # 可选，默认为 4096
CLAUDE_TIMEOUT="600"  # 可选，默认为 600 秒
```

### Gemini 模型

要使用 Gemini 模型，您需要：

1. 获取 Google API 密钥：访问 [Google AI Studio](https://makersuite.google.com/app/apikey) 注册并获取 API 密钥
2. 在 `.env` 文件中配置以下环境变量：

```
GOOGLE_API_KEY="your_google_api_key"
GEMINI_MODEL="gemini-1.5-pro"  # 可选，默认为 "gemini-1.5-pro"
GEMINI_TEMPERATURE="0"  # 可选，默认为 0
GEMINI_MAX_TOKENS="4096"  # 可选，默认为 4096
GEMINI_TIMEOUT="600"  # 可选，默认为 600 秒
```

## 使用示例

安装完成后，您可以使用以下命令来使用不同的模型：

```bash
# 使用 Claude 模型
python run_codedog.py repo-eval your-repo --model claude

# 使用 Gemini 模型
python run_codedog.py repo-eval your-repo --model gemini

# 使用单个提交评估模式（解决 token 限制问题）
python run_codedog.py repo-eval your-repo --model deepseek --batch-committer --single-commit
```

## 故障排除

### Token 限制问题

如果您遇到 token 限制错误（例如 "maximum context length exceeded"），可以尝试以下解决方案：

1. 使用 `--single-commit` 参数，每次只评估一个提交
2. 减小 `--model-token-limit` 参数的值（例如设置为 40000）
3. 使用具有更大上下文窗口的模型（Claude 或 Gemini）
4. 使用 `--include` 和 `--exclude` 参数限制评估范围

### 依赖冲突

如果您遇到依赖冲突，建议使用虚拟环境：

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 已知问题

#### modelcontextprotocol-github 包

`requirements.txt` 中的 `modelcontextprotocol-github` 包已被注释掉，因为它不在公共 PyPI 仓库中。这个包可能是项目特定的或内部使用的。

这个包主要用于 `fetch_samples_mcp.py` 示例脚本，该脚本用于从 GitHub 获取代码样本。这不是核心功能的一部分，所以您可以安全地忽略这个依赖，除非您特别需要运行该示例脚本。

如果您需要运行 `fetch_samples_mcp.py` 脚本，您可能需要：
1. 联系项目维护者获取 `modelcontextprotocol-github` 包的安装说明
2. 或者修改脚本以使用标准的 GitHub API（通过 PyGithub 包）来获取代码样本

## 更多信息

有关更多信息，请参阅：

- [Claude 文档](https://docs.anthropic.com/claude/docs)
- [Gemini 文档](https://ai.google.dev/docs)
- [CodeDog 文档](https://github.com/codedog-ai/codedog)
