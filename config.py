class Config(object):
    # 通用配置
    num_layers = 3
    data_path = 'data/'
    pickle_path = 'tang.npz'
    author = None
    constrain = None
    category = 'poet.tang'
    lr = 3e-4
    weight_decay = 1e-4
    use_gpu = True
    epoch = 30                 # 可以跑更多 epoch
    batch_size = 128            # 根据显存调整
    maxlen = 125               # 训练时的最大长度（原始数据）
    env = 'poetry'
    max_gen_len = 200          # 生成诗歌最长长度
    debug_file = '/tmp/debugp'
    model_path = None          # 开始训练时不加载预训练模型
    prefix_words = '仙路尽头谁为峰？一见无始道成空。'
    start_words = '闲云潭影日悠悠'
    acrostic = False
    model_prefix = 'checkpoints/tang_transformer'
    embedding_dim = 256
    hidden_dim = 512

    # ----- Transformer 专属配置 -----
    use_transformer = True
    num_heads = 8
    transformer_layers = 6
    dropout = 0.1
    warmup_steps = 2000
    transformer_max_len = 512   # 新增：Transformer 支持的最大序列长度（大于 maxlen）