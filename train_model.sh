#!/bin/bash

# 训练模型的脚本

# 检查数据是否存在，如果不存在则生成模拟数据
if [ ! -d "./data/train" ] || [ ! -d "./data/val" ] || [ ! -f "./data/stations.csv" ]; then
    echo "Training data not found. Generating mock data..."
    ./generate_data.sh
fi

# 设置参数
CONFIG_FILE="configs/low_memory.yaml"
CHECKPOINT_FILE=""  # 可以指定从某个检查点继续训练

# 激活conda环境（如果需要）
# conda activate pytorch

# 开始训练
echo "Starting model training..."
CMD="python -m src.main --mode train --config $CONFIG_FILE"
if [ -n "$CHECKPOINT_FILE" ]; then
    CMD="$CMD --checkpoint $CHECKPOINT_FILE"
fi

# 执行命令
eval $CMD

echo "Model training completed!"