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

训练过程中会保存模型检查点到 `checkpoints/` 目录，并在 `result_transformer_fixed.txt` 中记录训练损失和生成的示例诗歌。

### 测试生成

运行测试脚本：
```bash
python test.py
```

根据提示选择生成模式：
- 模式1：首句生成 - 输入诗歌的起始句子
- 模式2：藏头诗生成 - 输入藏头字（不超过16个字，建议偶数）

### 统一评估

不同模型的训练损失口径可能不同，尤其是padding处理不一致时，不能直接用loss判断诗歌质量。可以用统一评估脚本只评估生成文本：
```bash
python evaluate_poetry.py LSTM=result.txt TransformerOld=result_transformer.txt TransformerFixed=result_transformer_fixed.txt --output evaluation_report.csv
```

默认只评估每个日志文件最后一轮生成样例；如果要评估日志中的全部样例，加入 `--all`。

评估总分为0-100，由以下分项加权得到：
- `clean`: 是否泄漏 `<START>`、`<EOP>`、`</s>` 等特殊符号，以及是否出现语料外字符
- `form`: 句长是否接近五言/七言，标点密度、结尾和对句长度是否合理
- `fluency`: 重复字、重复2/3-gram、连续重复等现象越少越好
- `style`: 生成文本的字分布与唐诗语料的 Jensen-Shannon 相似度
- `length`: 生成长度是否接近训练语料长度分布
- `novelty`: 是否大量复刻训练语料中的长片段

## 文件说明

- `main.py` - 模型训练脚本，训练时会去掉数据左侧padding，并在batch内右侧补齐
- `model.py` - Transformer诗歌生成模型定义，保留旧LSTM模型类用于对照
- `generate.py` - 诗歌生成函数
- `test.py` - 交互式测试脚本
- `evaluate_poetry.py` - 统一文本质量评估脚本
- `config.py` - 配置文件，包含模型参数和训练设置
- `data.py` - 数据加载脚本
- `tang.npz` - 预处理好的唐诗数据集
- `checkpoints/` - 模型检查点保存目录
- `result_transformer_fixed.txt` - 修正padding后的Transformer训练结果记录文件

## 配置说明

主要配置参数在 `config.py` 中：

- `transformer_num_layers`: Transformer Encoder层数（默认6）
- `transformer_nhead`: 多头注意力头数（默认8）
- `transformer_dim_feedforward`: 前馈网络维度（默认1024）
- `transformer_dropout`: Dropout比例（默认0.1）
- `embedding_dim`: 词嵌入维度（默认256）
- `batch_size`: 批大小（默认128）
- `epoch`: 训练轮数（默认50）
- `lr`: 学习率（默认0.0003）
- `use_gpu`: 是否使用GPU（默认True）

修正padding后建议从头训练Transformer，将 `config.py` 中的 `model_path` 设为 `None` 即可；如果只是测试或继续训练，可将 `model_path` 指向已有检查点。

## 数据集

使用预处理好的唐诗数据集 `tang.npz`，包含：
- 诗歌文本数据
- 词到索引映射
- 索引到词映射

## 作者

研究生《高级人工智能》课程大作业

## 许可证

本项目仅用于学习和研究目的。
