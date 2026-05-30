# Poetry-Generation

研究生《高级人工智能》课程大作业

## 项目描述

这是一个基于深度学习的唐诗自动生成项目，使用PyTorch框架实现Transformer模型，通过训练唐诗数据集来生成新的诗歌。支持首句生成和藏头诗生成两种模式。

## 功能特性

- **诗歌生成**：基于给定的首句生成完整诗歌
- **藏头诗生成**：根据指定的藏头字生成诗歌
- **模型训练**：使用Transformer Encoder训练自回归诗歌生成模型
- **交互式测试**：提供命令行界面进行诗歌生成测试

## 环境要求

- Python 3.6+
- PyTorch 1.0+
- CUDA（可选，用于GPU加速）

## 依赖包

- torch
- numpy
- torchnet
- tqdm

## 安装步骤

1. 克隆或下载项目到本地
2. 安装依赖包：
   ```bash
   pip install torch numpy torchnet tqdm
   ```
3. 确保数据文件 `tang.npz` 存在于项目根目录

## 使用方法

### 训练模型

运行训练脚本：
```bash
python main.py
```

训练过程中会保存模型检查点到 `checkpoints/` 目录，并在 `result.txt` 中记录训练损失和生成的示例诗歌。

### 测试生成

运行测试脚本：
```bash
python test.py
```

根据提示选择生成模式：
- 模式1：首句生成 - 输入诗歌的起始句子
- 模式2：藏头诗生成 - 输入藏头字（不超过16个字，建议偶数）

## 文件说明

- `main.py` - 模型训练脚本
- `model.py` - Transformer诗歌生成模型定义，保留旧LSTM模型类用于对照
- `generate.py` - 诗歌生成函数
- `test.py` - 交互式测试脚本
- `config.py` - 配置文件，包含模型参数和训练设置
- `data.py` - 数据加载脚本
- `tang.npz` - 预处理好的唐诗数据集
- `checkpoints/` - 模型检查点保存目录
- `result.txt` - 训练结果记录文件

## 配置说明

主要配置参数在 `config.py` 中：

- `transformer_num_layers`: Transformer Encoder层数（默认6）
- `transformer_nhead`: 多头注意力头数（默认8）
- `transformer_dim_feedforward`: 前馈网络维度（默认1024）
- `transformer_dropout`: Dropout比例（默认0.1）
- `embedding_dim`: 词嵌入维度（默认256）
- `batch_size`: 批大小（默认256）
- `epoch`: 训练轮数（默认50）
- `lr`: 学习率（默认0.001）
- `use_gpu`: 是否使用GPU（默认True）

默认会加载 `checkpoints/tang_transformer_30.pth` 进行测试或继续训练；如果需要从头训练Transformer，将 `config.py` 中的 `model_path` 改为 `None`。

## 数据集

使用预处理好的唐诗数据集 `tang.npz`，包含：
- 诗歌文本数据
- 词到索引映射
- 索引到词映射

## 作者

研究生《高级人工智能》课程大作业

## 许可证

本项目仅用于学习和研究目的。
