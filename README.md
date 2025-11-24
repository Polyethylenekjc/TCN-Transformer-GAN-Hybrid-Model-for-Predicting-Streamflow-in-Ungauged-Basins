# 时空径流预测系统 (Temporal Streamflow Forecasting System)

端到端的径流预测模型，输入多日多通道图像和站点数据，输出下一日高分辨率径流预测图像。

## 系统特性

✅ **模块化结构** - CNN编码器 + 时间建模(LSTM/Transformer) + 上采样解码器
✅ **参数自适应** - 自动推断维度，防止维度不匹配错误
✅ **配置集中化** - 统一YAML管理所有参数
✅ **防OOM优化** - 支持混合精度、梯度累积、补丁训练
✅ **多任务学习** - 图像预测 + 站点值预测
✅ **灵活时间建模** - LSTM与Transformer可互换

## 项目结构

```
project_root/
├── data/
│   ├── images/              # 输入图像 (timestamp.npy)
│   ├── stations/            # 站点数据 (station_name.csv)
│   └── config.yaml          # 配置文件
├── src/
│   ├── dataset.py           # 数据集加载与处理
│   ├── model/
│   │   ├── cnn_encoder.py      # CNN空间特征提取
│   │   ├── lstm_module.py      # LSTM时间建模
│   │   ├── transformer_module.py # Transformer时间建模
│   │   ├── decoder.py          # 上采样解码器
│   │   ├── gan.py              # GAN模块 (可选)
│   │   └── __init__.py         # 完整模型定义
│   ├── loss.py              # 损失函数 (图像+站点+GAN)
│   ├── train.py             # 训练脚本
│   ├── evaluate.py          # 评估脚本
│   └── utils/
│       ├── config_loader.py    # YAML配置加载
│       └── param_autoalign.py  # 参数自动对齐
├── generate_dummy_data.py   # 生成虚拟数据
├── main.py                  # 主程序入口
├── requirements.txt         # 依赖
└── README.md                # 本文档
```

## 常用指令 (Common Commands)

### 1. 数据准备 (Data Preparation)

下载 ERA5 和 GLOFAS 数据并合并为 `.npy` 格式：
```bash
# 下载并处理 2015-2024 年的数据
python src/utils/data_download.py --start-year 2015 --end-year 2024
```

### 2. 训练模型 (Training)

**从头开始训练：**
```bash
python src/train.py --config data/config.yaml
```

**中断后继续训练 (Resume)：**
```bash
# 从最佳模型继续训练 50 个 epoch
python src/train.py --resume output/best_model.pt --epochs 50
```

**指定 Epoch 数：**
```bash
python src/train.py --epochs 200
```

### 3. 评估模型 (Evaluation)

评估模型在测试集上的表现：
```bash
python src/evaluate.py --model output/best_model.pt
```

### 4. 可视化预测结果 (Visualization)

生成预测结果的对比图（支持反归一化显示真实数值）：
```bash
# 可视化 20200101 的预测结果，并反归一化
python src/utils/visualize_predictions.py --date 20200101 --model output/best_model.pt --denormalize
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 生成虚拟数据用于快速验证

```bash
python main.py generate --days 30 --stations 5 --channels 10
```

参数说明：
- `--days 30` - 生成30天的数据
- `--stations 5` - 生成5个站点的数据
- `--channels 10` - 图像10个通道

生成文件：
- `data/images/20200101.npy` - (C, H, W) 多通道图像（H,W 由 `data.image_size` 配置）
- `data/stations/station_00.csv` - 站点数据
- `data/config.yaml` - 配置文件

### 3. 训练模型

```bash
python main.py train --config data/config.yaml
```

训练特性：
- 自动使用CUDA (如果可用)
- 支持混合精度加速
- 梯度累积防显存溢出
- 每个epoch保存最佳模型
- 实时GPU显存日志

### 4. 评估模型

```bash
python main.py evaluate --config data/config.yaml --model output/best_model.pt
```

输出：
- `output/predictions/` - 预测的径流图
- `output/stations_eval.csv` - 站点预测对比
- 指标：RMSE, MAE, R², NSE

## 配置文件详解 (config.yaml)

```yaml
data:
  image_dir: ./data/images          # 输入图像目录
  station_dir: ./data/stations      # 站点数据目录
  output_dir: ./output              # 输出目录
  resolution: 0.05                  # 地理分辨率 (度)
  region: [100, 110, 25, 35]        # 地理范围 [lon_min, lon_max, lat_min, lat_max]
  window_size: 5                    # 时间窗口大小
  upscale_factor: 2                 # 上采样倍数 (2表示分辨率翻倍)

model:
  input_channels: 10                # 输入图像通道数
  hidden_dim: 128                   # 隐层维度
  num_layers: 2                     # 模型层数
  backbone: CNN                     # 主干网络
  temporal_module: LSTM             # 时间建模 ('LSTM' 或 'Transformer')
  use_gan: false                    # 是否使用GAN

train:
  batch_size: 4                     # 批处理大小
  lr: 1e-4                          # 学习率
  num_epochs: 100                   # 训练轮数
  device: cuda                      # 设备 ('cuda' 或 'cpu')

loss:
  metric: MSE                       # 损失函数类型
  weights:
    image: 1.0                      # 图像损失权重
    station: 0.5                    # 站点损失权重

stations:
  kernel_size: 3                    # 站点邻域窗口
  metric: MSE                       # 站点评估指标

system:
  mixed_precision: true             # 混合精度 (AMP)
  grad_accum_steps: 2               # 梯度累积步数
  log_interval: 10                  # 日志间隔
```

## 数据格式说明

### 输入图像 (data/images/)

文件名: `YYYYMMDD.npy` (e.g., `20200101.npy`)

格式: NumPy数组, 形状 `(C, H, W)` 例如 `(10, H, W)`（H,W 由 `data.image_size` 配置，默认 `[128, 128]`）
- C: 通道数 (气象变量，由`config.yaml` 的 `model.input_channels` 控制)
- H, W: 空间分辨率（由 `data.image_size` 控制，默认 128×128）

```python
import numpy as np
image = np.random.randn(10, H, W)  # 10通道图像 (H,W 来自 `data.image_size`)
np.save('20200101.npy', image)
```

### 站点数据 (data/stations/)

文件名: `{station_name}.csv` (e.g., `station_00.csv`)

CSV格式，包含列：
```
timestamp,lon,lat,runoff
20200101,105.5,30.2,120.5
20200102,105.5,30.2,135.2
...
```

- `timestamp`: 日期 (YYYYMMDD格式)
- `lon`: 经度 (度)
- `lat`: 纬度 (度)
- `runoff`: 径流值 (立方米/秒或其他单位)

## 模型架构

### 1. CNN编码器 (空间特征提取)

```
输入: (B, T, C, H, W)
  ↓
重塑: (B*T, C, H, W)
  ↓
多层卷积+BatchNorm+ReLU
  ↓
MaxPooling (特征图缩小)
  ↓
输出: (B*T, out_ch, h', w')
```

自动特性：
- 逐层倍增通道数
- 自动计算输出维度
- 支持configurable层数和kernel大小

### 2. 时间建模模块 (可互换)

#### LSTM (默认)

```
输入: (B, T, C*H*W)
  ↓
可选线性映射到hidden_dim
  ↓
LSTM (可堆叠多层)
  ↓
输出: 最后隐藏状态 (B, hidden_dim)
```

#### Transformer (可选)

```
输入: (B, T, C*H*W)
  ↓
线性投影 → hidden_dim
  ↓
位置编码
  ↓
Transformer编码器 (多层)
  ↓
时间平均池化
  ↓
输出: (B, hidden_dim)
```

**切换方法**：修改config.yaml中的`model.temporal_module: Transformer`

### 3. 上采样解码器

```
输入: (B, hidden_dim, h', w')
  ↓
投影回空间特征 (B, out_ch, h', w')
  ↓
卷积+ReLU
  ↓
PixelShuffle或插值上采样 (×2)
  ↓
refine卷积
  ↓
输出: (B, 1, H_out, W_out)
```

输出分辨率 = 输入 × upscale_factor

### 4. GAN模块 (可选)

- **生成器**: CNN + LSTM/Transformer + 解码器
- **判别器**: PatchGAN (4层卷积)
- **损失**: 对抗损失 + 像素损失 (L1)

启用：`model.use_gan: true` 在config.yaml

## 损失函数

### 总损失 = 图像损失 + 站点损失 + GAN损失

```
L_total = w_img * L_img + w_stn * L_stn + w_gan * L_gan
```

### 1. 图像损失

逐像素MSE或MAE：
```
L_img = ||prediction - target||₂²
```

### 2. 站点损失

基于邻域的站点值预测损失：
```
对每个站点：
  1. 将lon/lat映射到像素位置(px, py)
  2. 提取3×3或5×5邻域
  3. 计算平均预测值 vs 真实runoff
  4. 求MSE/MAE
```

### 3. GAN损失 (可选)

```
L_gen = λ₁ * L_adversarial + λ₂ * L_pixel
L_disc = L_real + L_fake
```

## 防OOM优化策略

### 1. 混合精度 (Mixed Precision)

```yaml
system:
  mixed_precision: true
```

- 自动使用float16进行前向计算
- float32维护模型精度
- 显存节省30-40%

### 2. 梯度累积 (Gradient Accumulation)

```yaml
system:
  grad_accum_steps: 2
```

有效批大小 = batch_size × grad_accum_steps
- 显存占用 ↓
- 每2步更新一次模型参数

### 3. 显存日志

每10个梯度更新步打印：
```
GPU Memory - Allocated: 2345.6MB, Reserved: 2816.0MB
```

### 4. 推荐配置 (12GB显存)

```yaml
train:
  batch_size: 4
  device: cuda

system:
  mixed_precision: true
  grad_accum_steps: 2
```

## 训练过程

### 单个epoch的流程

```
1. 加载批数据 (B, T, C, H, W) 和 (B, 1, H_out, W_out)
2. CNN编码: (B*T, C, H, W) → (B*T, out_ch, h, w)
3. 时间建模: (B, T, out_ch*h*w) → (B, hidden_dim)
4. 空间恢复: (B, hidden_dim) → (B, out_ch, h, w)
5. 解码上采样: (B, out_ch, h, w) → (B, 1, H_out, W_out)
6. 计算损失: L = w_img*L_img + w_stn*L_stn
7. 反向传播 + 优化器更新
8. 梯度累积 (如配置)
```

### 每个epoch的输出

```
Epoch 0 Batch 10: Loss=0.234567, Img=0.200000, Stn=0.006789
Epoch 0 - Train Loss: 0.198765
Epoch 0 - Val Loss: 0.187654, RMSE: 1.2345, NSE: 0.8765
GPU Memory - Allocated: 2345.6MB, Reserved: 2816.0MB
```

## 评估指标

### 图像级别指标

- **RMSE** (Root Mean Squared Error)
  ```
  RMSE = sqrt(mean((pred - target)²))
  ```
  
- **MAE** (Mean Absolute Error)
  ```
  MAE = mean(|pred - target|)
  ```

- **R²** (决定系数)
  ```
  R² = 1 - SS_res / SS_tot
  ```
  范围 [0, 1]，越高越好

- **NSE** (Nash-Sutcliffe Efficiency)
  ```
  NSE = 1 - sum((obs - pred)²) / sum((obs - mean(obs))²)
  ```
  范围 (-∞, 1]，常用于水文评估

### 站点级别评估

- 位置精度: 站点坐标到像素的映射
- 邻域平均: 3×3或5×5窗口内的平均预测
- 逐站点RMSE/NSE

## 高级用法

### 1. 从检查点恢复训练

```python
from src.train import Trainer
from src.utils.config_loader import ConfigLoader

config = ConfigLoader.load_config('data/config.yaml')
trainer = Trainer(config, model_path='./output/checkpoint_epoch_50.pt')
trainer.train(dataset, num_epochs=100)
```

### 2. 自定义数据加载

```python
from src.dataset import StreamflowDataset

dataset = StreamflowDataset(
    image_dir='./data/images',
    station_dir='./data/stations',
    config=config,
    normalize=True  # 自动标准化
)

# 访问单个样本
sample = dataset[0]
print(sample.keys())  # images, output_image, stations, station_positions
```

### 3. 修改模型架构

编辑 `src/model/__init__.py` 中的 `StreamflowPredictionModel.__init__()`:

```python
# 改变CNN层数
self.cnn_encoder = CNNEncoder(
    input_channels=self.input_channels,
    hidden_dim=self.hidden_dim,
    num_layers=4,  # ← 从2改为4
    kernel_size=5   # ← 调整kernel大小
)

# 改变LSTM为Transformer
if self.temporal_module_type.upper() == 'TRANSFORMER':
    self.temporal_module = TransformerModule(
        input_dim=cnn_feature_size,
        hidden_dim=self.hidden_dim,
        num_layers=4,        # ← 更多层
        num_heads=8          # ← 更多注意力头
    )
```

### 4. 批量预测

```python
from src.evaluate import Evaluator
import torch

evaluator = Evaluator(config, 'output/best_model.pt')

# 生成多个样本
for batch_images in data_loader:
    predictions = evaluator.predict_batch(batch_images)  # (B, 1, 256, 256)
    # 后续处理...
```

## 故障排查

### 问题1: OOM错误

**症状**: `RuntimeError: CUDA out of memory`

**解决**:
```yaml
# 降低批大小
train:
  batch_size: 2

# 增加梯度累积
system:
  grad_accum_steps: 4

# 启用混合精度
system:
  mixed_precision: true
```

### 问题2: 维度不匹配

**症状**: `RuntimeError: Dimension mismatch at XXX`

**原因**: 图像大小与预期不符

**解决**:
- 检查数据: `np.load('data/images/20200101.npy').shape` 应为 `(C, H, W)`，其中 H/W 对应 `data.image_size` 中设置的值
- 确保所有图像尺寸一致
- 检查config.yaml中的upscale_factor

### 问题3: 训练不收敛

**症状**: 损失函数不下降

**检查**:
1. 学习率是否过高/过低
   ```yaml
   train:
     lr: 5e-5  # 尝试降低
   ```

2. 数据是否标准化
   ```python
   dataset = StreamflowDataset(..., normalize=True)
   ```

3. 数据量是否足够
   ```bash
   python main.py generate --days 100  # 增加天数
   ```

### 问题4: 站点损失为0

**症状**: 训练log显示 `Stn=0.000000`

**原因**: CSV文件中的timestamp列格式不匹配

**检查**:
```python
df = pd.read_csv('data/stations/station_00.csv')
print(df.head())  # 检查timestamp格式是否为YYYYMMDD
```

## 性能基准 (参考值)

在NVIDIA RTX 3090 (24GB) 上，100个epoch的训练时间：

| 配置 | 显存占用 | 每epoch耗时 |
|------|---------|-----------|
| 原始 (batch=4) | 18.5GB | 2.3s |
| AMP (mixed_precision=true) | 12.1GB | 1.8s |
| Grad累积 (x4) | 8.5GB | 4.2s |
| AMP + Grad累积 | 6.2GB | 3.5s |

## 参数自动计算示例

系统自动处理以下计算 (无需手动干预):

```python
# CNN输出维度自动计算
cnn_output_shape = (256, 32, 32)  # 基于num_layers和kernel_size

# LSTM/Transformer维度自动匹配
temporal_input_dim = 256 * 32 * 32  # 自动展平

# 上采样输出尺寸自动计算
output_shape = (1, 256, 256)  # 128 * 2 = 256

# 空间恢复线性层自动初始化
spatial_projection = Linear(hidden_dim, 256*32*32)
```

## 扩展建议

1. **多任务学习**: 同时预测多个物理量
   - 修改decoder输出通道数
   
2. **注意力机制**: 引入空间注意力
   - 在CNN编码器后添加CBAM模块
   
3. **集合方法**: 多模型ensemble
   - 训练多个config变体
   - 加权平均预测

4. **实时预报**: 流式处理
   - 修改dataset为时间序列预报模式
   - 使用滑动窗口进行在线预测

## 许可证

MIT License

## 联系方式

如有问题，请检查:
1. 日志文件: `output/training_*.log`
2. 验证数据格式
3. 确认config.yaml参数正确
