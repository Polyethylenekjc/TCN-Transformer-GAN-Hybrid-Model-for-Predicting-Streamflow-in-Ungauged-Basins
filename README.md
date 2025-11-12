# TCN-Transformer-Diffusion-Hybrid-Model-for-Predicting-Streamflow-in-Ungauged-Basins

本项目实现了一个混合模型，用于在无测量流域(Ungauged Basins)中进行精确的径流预测。

## 架构设计

系统采用以下架构：

1. **Spatial CNN**: 提取流域的空间特征（如地形、土壤类型、土地利用等）
2. **Temporal TCN**: 使用TCN学习径流的时间序列变化规律
3. **Global Transformer**: 使用Transformer学习不同流域间的全局上下文依赖
4. **Conditional DDPM/UNet**: 条件扩散模型生成目标流域的径流预测

## 项目结构

```
src/
├── models/                 # 模型组件
│   ├── spatial_encoder.py   # 空间特征提取器
│   ├── temporal_encoder.py  # 时序编码器
│   ├── transformer_encoder.py # Transformer编码器
│   ├── conditional_unet.py  # 条件扩散模型
│   └── full_model.py        # 完整模型整合
├── data/                   # 数据处理
│   └── preprocessing.py     # 数据预处理和增强
├── utils/                  # 工具函数
│   ├── losses.py           # 损失函数
│   └── metrics.py          # 评估指标
├── train/                  # 训练相关
│   └── trainer.py          # 训练器
└── main.py                 # 主程序入口
```

## 核心组件说明

### 1. Spatial Encoder (CNN)
对每个流域的空间特征图（如DEM、土壤类型、植被覆盖等）进行特征提取。

### 2. Temporal TCN
使用因果膨胀卷积处理气象强迫数据（降水、温度等）的时间序列，捕获短期和中期依赖关系。

### 3. Transformer Encoder
学习跨流域的大尺度空间依赖和注意力权重，实现知识从有测量流域到无测量流域的迁移。

### 4. Conditional UNet for Diffusion
基于条件的扩散模型生成连续且准确的径流时间序列。

## 使用方法

### 安装依赖
```bash
pip install torch torchvision tqdm numpy pandas
```

### 训练模型
```bash
python src/main.py --mode train
```

### 生成预测
```bash
python src/main.py --mode sample
```

## 配置参数

主要超参数建议：
- SpatialEncoder 输出通道: d_model = 64 或 128
- Transformer 层数: 4，头数: 8
- TCN 层数: 4，膨胀系数: [1,2,4,8]
- 扩散步数: 训练1000，采样50-200步
- 批次大小: 8-32（根据显存调整）
- 学习率: 1e-4 (AdamW)

## 损失函数

1. **扩散损失**: MSE between true noise and predicted noise
2. **径流预测损失**: NSE（Nash-Sutcliffe Efficiency）或KGE（Kling-Gupta Efficiency）
3. **区域适应损失**: 用于减少源流域和目标流域之间的特征分布差异

总损失: L = L_diff + λ_pred*L_pred + λ_adapt*L_adapt

## 数据预处理

- 输入: 多个流域的历史气象 forcing 数据和静态属性
- 标签: 目标流域的径流时间序列
- 归一化: 每个流域的特征分别做标准化处理
- 数据增强: 在有测量流域上进行元学习训练

## 训练流程

1. 从数据加载器获取批次数据（支持多个流域）
2. SpatialEncoder对每个流域的空间特征进行编码
3. TCN处理气象 forcing 数据的时间序列
4. Transformer实现跨流域的知识迁移
5. 随机采样时间步，添加噪声
6. UNet预测噪声并计算损失
7. 反向传播更新参数

## 推理流程

1. 给定目标流域的空间特征和气象 forcing 数据
2. 获取条件编码
3. 从噪声开始逐步去噪生成最终的径流预测
4. 可使用分类器自由引导提高预测稳定性

# TTF - Time Series Forecasting with Transformers and Diffusion

This project implements a time series forecasting model using transformers and diffusion models for streamflow prediction in ungauged basins.

## Project Structure

```
src/
├── data/              # Data preprocessing modules
├── models/            # Model definitions
├── train/             # Training utilities
├── utils/             # Utility functions
└── main.py           # Main entry point
configs/               # Configuration files
```

## Getting Started

### Prerequisites

- Python 3.7+
- PyTorch 1.8+
- Other dependencies in `requirements.txt`

### Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training

```bash
python -m src.main --mode train --config configs/default.yaml
```

### Sampling

```bash
python -m src.main --mode sample --config configs/default.yaml
```

## Configuration

The project supports configuration files to adjust model parameters. Example configuration files are provided in the `configs/` directory:

- `configs/default.yaml`: Default model configuration
- `configs/small_model.yaml`: Smaller model for testing
- `configs/large_model.yaml`: Larger model for better performance

### Configuration Options

- `model.input_channels`: Number of input channels
- `model.d_model`: Model dimension
- `model.tcn_dilations`: Dilations for TCN layers
- `model.transformer_num_heads`: Number of attention heads in transformer
- `model.transformer_num_layers`: Number of transformer layers
- `model.diffusion_time_steps`: Number of diffusion time steps
- `training.learning_rate`: Learning rate for training
- `training.batch_size`: Training batch size
- `training.epochs`: Number of training epochs
- `hardware.device`: Device to run on ("cuda" or "cpu")
- `data.sequence_length`: Length of input sequence
- `data.input_height`: Height of input images
- `data.input_width`: Width of input images

## Model Architecture

The model consists of four main components:
1. Spatial Encoder (CNN) - Extracts spatial features from basin characteristics
2. Temporal Encoder (TCN) - Processes meteorological forcing data
3. Global Context Encoder (Transformer) - Enables knowledge transfer between gauged and ungauged basins
4. Diffusion Decoder (Conditional UNet) - Generates accurate streamflow predictions

## License

This project is licensed under the MIT License - see the LICENSE file for details.