

class Config(object):
    model_type = 'transformer'  # 当前默认使用Transformer架构
    num_layers = 3  # LSTM层数，保留用于加载旧模型
    data_path = 'data/'  # 诗歌的文本文件存放路径
    pickle_path = 'tang.npz'  # 预处理好的二进制文件
    author = None  # 只学习某位作者的诗歌
    constrain = None  # 长度限制
    category = 'poet.tang'  # 类别，唐诗还是宋诗歌(poet.song)
    lr = 1e-3
    weight_decay = 1e-4
    use_gpu = True
    epoch = 50
    batch_size = 256
    maxlen = 125  # 超过这个长度的之后字被丢弃，小于这个长度的在前面补空格
    # use_env = True # 是否使用visodm
    env = 'poetry'  # visdom env
    max_gen_len = 200  # 生成诗歌最长长度
    debug_file = '/tmp/debugp'
    model_path = "./checkpoints/tang_transformer_30.pth"  # 预训练模型路径，设为None可从头训练
    prefix_words = '仙路尽头谁为峰？一见无始道成空。'  # 不是诗歌的组成部分，用来控制生成诗歌的意境
    start_words = '闲云潭影日悠悠'  # 诗歌开始
    acrostic = False  # 是否是藏头诗
    model_prefix = 'checkpoints/tang_transformer_run'  # 模型保存路径
    result_path = 'result_transformer.txt'
    embedding_dim = 256
    hidden_dim = 512  # LSTM隐藏层维度，Transformer默认不使用
    transformer_nhead = 8
    transformer_num_layers = 6
    transformer_dim_feedforward = 1024
    transformer_dropout = 0.1
    max_position_len = 512
