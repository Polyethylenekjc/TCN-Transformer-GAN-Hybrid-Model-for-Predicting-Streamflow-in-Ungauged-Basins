#!/bin/bash

# 生成模拟训练数据的脚本

echo "Generating mock training data..."

# 激活conda环境（如果需要）
# conda activate pytorch

# 生成模拟数据
python -m src.generate_mock_data

echo "Mock data generation completed!"