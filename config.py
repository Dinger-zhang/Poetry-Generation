# config.py

class Config(object):
    # ========== 通用配置 ==========
    use_gpu = True
    device = None                     # 运行时自动设置
    epoch = 30
    batch_size = 16                    # GPT‑2 显存占用大，batch 需调小
    max_gen_len = 200
    model_path = None                 # 预训练模型路径（用于继续训练）

    # ========== 数据相关 ==========
    data_path = 'data/'
    pickle_path = 'tang.npz'

    # ========== 原有模型（LSTM / Transformer）配置 ==========
    use_transformer = False           # 改用 GPT‑2 时设为 False
    num_layers = 3
    embedding_dim = 256
    hidden_dim = 512
    num_heads = 8
    transformer_layers = 6
    dropout = 0.1
    warmup_steps = 2000
    transformer_max_len = 512

    # ========== GPT‑2 微调配置 ==========
    use_pretrained_gpt2 = True        # 启用 GPT‑2 预训练模型
    gpt2_model_name = "xingyu1996/chinese-poems-gpt2"
    gpt2_max_length = 128             # 训练/生成时最大长度（不超过模型支持长度）
    gpt2_learning_rate = 5e-5
    gpt2_weight_decay = 0.01
    gpt2_warmup_steps = 500
    gpt2_logging_steps = 100
    gpt2_save_steps = 500
    gpt2_output_dir = "checkpoints/gpt2_poetry"
    gpt2_overwrite_output_dir = True
    gpt2_fp16 = False                 # 若 GPU 支持可开启