# 代码评审提示测试用例

这个目录包含了用于测试代码评审提示(prompts)效果的diff文件集合。这些文件被分为三类，以便测试提示在不同质量代码上的表现。

## 目录结构

```
test_diffs/
├── high_score/     # 高质量代码变更，应该获得高分
├── low_score/      # 低质量代码变更，应该获得低分
└── mixed_score/    # 混合质量代码变更，应该获得中等分数
```

## 高分测试用例 (high_score/)

这些diff文件展示了高质量的代码变更，包括良好的文档、错误处理、性能优化和安全实践。它们应该在代码评审中获得高分。

1. **python_feature_enhancement.diff** - Python文件的功能增强，添加了类型提示、错误处理和新功能
2. **javascript_bug_fix.diff** - JavaScript文件的bug修复，改进了错误处理和安全性
3. **java_refactoring.diff** - Java文件的代码重构，改进了架构和设计模式
4. **css_optimization.diff** - CSS文件的样式优化，使用了CSS变量和现代布局技术
5. **sql_query_optimization.diff** - SQL查询优化，提高了查询性能和增加了分析功能

## 低分测试用例 (low_score/)

这些diff文件展示了低质量的代码变更，包含各种问题，如安全漏洞、性能问题、错误处理不足和可读性差。它们应该在代码评审中获得低分。

1. **python_security_issues.diff** - 包含多个安全漏洞的Python代码，如SQL注入、命令注入和硬编码凭据
2. **javascript_performance_issues.diff** - 包含性能问题的JavaScript代码，如内存泄漏、低效循环和DOM操作
3. **java_error_handling_issues.diff** - 错误处理不足的Java代码，如异常吞没、资源泄漏和线程不安全
4. **cpp_readability_issues.diff** - 可读性差的C++代码，包含混乱的格式、神秘的变量名和缺乏注释
5. **sql_structure_issues.diff** - 结构混乱的SQL代码，包含不一致的格式、低效查询和缺乏注释

## 使用方法

### 单个文件测试

使用`test_prompt.py`工具测试单个diff文件：

```bash
python test_prompt.py --diff test_diffs/high_score/python_feature_enhancement.diff --model gpt-3.5-turbo
```

### 批量测试

使用`batch_test_prompts.py`工具批量测试多个diff文件：

```bash
# 测试所有高分用例
for diff_file in test_diffs/high_score/*.diff; do
  python test_prompt.py --diff "$diff_file" --model gpt-3.5-turbo --output "${diff_file%.diff}_result.json"
done

# 测试所有低分用例
for diff_file in test_diffs/low_score/*.diff; do
  python test_prompt.py --diff "$diff_file" --model gpt-3.5-turbo --output "${diff_file%.diff}_result.json"
done
```

### 比较不同模型

比较不同模型在同一组测试用例上的表现：

```bash
python batch_test_prompts.py --project your_group/your_project --models gpt-3.5-turbo,gpt-4,deepseek
```

## 评估标准

使用这些测试用例评估代码评审提示时，应关注以下方面：

1. **分数准确性** - 高分用例应获得高分，低分用例应获得低分
2. **问题识别** - 是否正确识别代码中的问题和优点
3. **有效代码识别** - 是否正确区分有效和无效代码修改
4. **工作时间估算** - 工作时间估算是否合理
5. **建议质量** - 提供的改进建议是否具体、可行

通过比较不同提示和模型在这些测试用例上的表现，可以找到最适合您需求的组合。
