#!/bin/bash
# Quick start script for Streamflow Prediction System

set -e

echo "=========================================="
echo "时空径流预测系统 - 快速启动"
echo "=========================================="
echo ""

# 检查Python版本
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "✓ Python版本: $python_version"

# 检查依赖
echo ""
echo "检查依赖..."
python -c "import torch; print(f'✓ PyTorch: {torch.__version__}')"
python -c "import numpy; print(f'✓ NumPy: {numpy.__version__}')"
python -c "import pandas; print(f'✓ Pandas: {pandas.__version__}')"
python -c "import yaml; print(f'✓ PyYAML')"

# 生成虚拟数据
echo ""
echo "生成虚拟数据..."
python generate_dummy_data.py --days 30 --stations 5 --channels 10 --output-dir ./data

# 显示配置
echo ""
echo "配置概览:"
python -c "
import yaml
with open('./data/config.yaml') as f:
    config = yaml.safe_load(f)
print(f'  输入通道: {config[\"model\"][\"input_channels\"]}')
print(f'  隐层维度: {config[\"model\"][\"hidden_dim\"]}')
print(f'  时间模块: {config[\"model\"][\"temporal_module\"]}')
print(f'  训练批大小: {config[\"train\"][\"batch_size\"]}')
print(f'  学习率: {config[\"train\"][\"lr\"]}')
print(f'  GPU: {config[\"train\"][\"device\"]}')
"

echo ""
echo "=========================================="
echo "✅ 准备就绪！"
echo ""
echo "接下来的步骤:"
echo ""
echo "  1. 训练模型:"
echo "     python main.py train --config data/config.yaml"
echo ""
echo "  2. 评估模型:"
echo "     python main.py evaluate --config data/config.yaml --model output/best_model.pt"
echo ""
echo "  3. 修改配置:"
echo "     编辑 data/config.yaml"
echo "     例如: temporal_module: Transformer (切换到Transformer)"
echo ""
echo "  4. 查看文档:"
echo "     查看 README.md 了解详细信息"
echo ""
echo "=========================================="
