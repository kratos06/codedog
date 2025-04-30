import os
import sys
import importlib.util

# 加载 code_evaluator.py 模块
spec = importlib.util.spec_from_file_location("code_evaluator", "codedog/utils/code_evaluator.py")
code_evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(code_evaluator)

# 创建一个简单的类来测试 _estimate_default_hours 方法
class TestEstimator:
    def __init__(self):
        pass
        
    def _estimate_default_hours(self, additions, deletions, file_path=None):
        # 简单的启发式方法:
        # - 每50行添加大约需要1小时的时间（对于有经验的开发者）
        # - 每100行删除大约需要0.5小时的时间
        # - 最小0.5小时，最大40小时（1周）
        estimated_hours = (additions / 50) + (deletions / 200)
        return max(0.5, min(40, round(estimated_hours, 1)))

# 创建一个新的估算器，使用我们修改后的方法
class NewEstimator:
    def __init__(self):
        pass
        
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

# 测试不同的文件和行数
test_cases = [
    (5, 0, "small_file.py"),
    (10, 0, "small_file.py"),
    (20, 0, "medium_file.py"),
    (50, 0, "medium_file.py"),
    (100, 0, "large_file.py"),
    (190, 0, "test_complex.py"),
    (190, 0, "core/engine.py"),
    (190, 0, "utils/helper.py"),
    (190, 0, "test/test_file.py"),
    (190, 0, "complex.cpp"),
    (190, 0, "simple.html"),
    (500, 0, "huge_file.py"),
    (1000, 0, "massive_file.py"),
]

# 创建估算器
old_estimator = TestEstimator()
new_estimator = NewEstimator()

# 运行测试
print("Testing working hour estimation:")
print("=" * 80)
print(f"{'File':<20} {'Lines':<10} {'Old Hours':<10} {'New Hours':<10} {'Difference':<10}")
print("-" * 80)

for additions, deletions, file_path in test_cases:
    old_hours = old_estimator._estimate_default_hours(additions, deletions)
    new_hours = new_estimator._estimate_default_hours(additions, deletions, file_path)
    diff = new_hours - old_hours
    print(f"{file_path:<20} {additions+deletions:<10} {old_hours:<10.1f} {new_hours:<10.1f} {diff:+.1f}")

print("=" * 80)
