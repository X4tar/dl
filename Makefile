.PHONY: install dev-install test run-example run-tests clean format lint help

# 默认目标
help:
	@echo "可用的命令:"
	@echo "  install      - 安装项目依赖"
	@echo "  dev-install  - 安装开发依赖"
	@echo "  test         - 运行测试"
	@echo "  run-example  - 运行基本示例"
	@echo "  run-demo     - 运行完整演示"
	@echo "  run-tests    - 运行所有测试"
	@echo "  format       - 格式化代码"
	@echo "  lint         - 代码检查"
	@echo "  clean        - 清理生成的文件"

# 安装基础依赖
install:
	uv sync

# 安装开发依赖
dev-install:
	uv sync --all-extras

# 运行基本示例
run-example:
	uv run python cbow_model.py

# 运行完整演示
run-demo:
	uv run python cbow_example.py

# 运行测试
test:
	uv run python test_cbow.py

# 运行所有测试
run-tests: test

# 格式化代码
format:
	uv run black *.py

# 代码检查
lint:
	uv run flake8 *.py

# 清理生成的文件
clean:
	rm -f *.pth
	rm -f *.png
	rm -rf __pycache__
	rm -rf .pytest_cache
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
