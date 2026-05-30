import os
import torch as t
import numpy as np
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
from torchnet import meter
from tqdm import tqdm
from config import Config
from generate import generate

if Config.use_transformer:
    from model_transformer import TransformerPoetryModel as PoetryModel
else:
    from model import PoetryModel

def train():
    os.makedirs('checkpoints', exist_ok=True)

    if Config.use_gpu:
        Config.device = t.device("cuda")
    else:
        Config.device = t.device("cpu")
    device = Config.device

    # 加载数据
    datas = np.load("tang.npz", allow_pickle=True)
    data = datas['data']
    ix2word = datas['ix2word'].item()
    word2ix = datas['word2ix'].item()
    data = t.from_numpy(data)
    dataloader = DataLoader(data,
                            batch_size=Config.batch_size,
                            shuffle=True,
                            num_workers=4,
                            pin_memory=True)

    # 初始化模型
    vocab_size = len(word2ix)
    if Config.use_transformer:
        model = PoetryModel(vocab_size,
                            embedding_dim=Config.embedding_dim,
                            num_heads=Config.num_heads,
                            num_layers=Config.transformer_layers,
                            max_len=Config.maxlen,
                            dropout=Config.dropout)
    else:
        model = PoetryModel(vocab_size,
                            embedding_dim=Config.embedding_dim,
                            hidden_dim=Config.hidden_dim)

    optimizer = optim.AdamW(model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # 学习率预热
    if Config.use_transformer and Config.warmup_steps > 0:
        from torch.optim.lr_scheduler import LambdaLR
        def lambda_lr(step):
            if step < Config.warmup_steps:
                return step / Config.warmup_steps
            else:
                return 1.0
        scheduler = LambdaLR(optimizer, lr_lambda=lambda_lr)
    else:
        scheduler = None

    # 加载预训练模型（如果有）
    if Config.model_path and os.path.exists(Config.model_path):
        state_dict = t.load(Config.model_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained model from {Config.model_path}")
    else:
        print("No pretrained model found, starting from scratch.")

    model.to(device)
    loss_meter = meter.AverageValueMeter()

    f = open('result_transformer.txt', 'w', encoding='utf-8') if Config.use_transformer else open('result.txt', 'w', encoding='utf-8')

    for epoch in range(Config.epoch):
        loss_meter.reset()
        model.train()
        for li, data_ in tqdm(enumerate(dataloader), total=len(dataloader)):
            data_ = data_.long().transpose(1, 0).contiguous()   # (seq_len, batch)
            data_ = data_.to(device)
            optimizer.zero_grad()

            input_ = data_[:-1, :]      # (seq_len-1, batch)
            target = data_[1:, :]       # (seq_len-1, batch)

            output, _ = model(input_)
            loss = criterion(output.view(-1, output.size(-1)), target.view(-1))
            loss.backward()

            t.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            loss_meter.add(loss.item())

        avg_loss = loss_meter.mean
        print(f"Epoch {epoch+1}/{Config.epoch} 训练损失为 {avg_loss:.6f}")
        f.write(f"Epoch {epoch+1} 训练损失为 {avg_loss:.6f}\n")
        f.flush()

        # 每个 epoch 后生成示例诗歌（限制生成长度，避免超出模型范围）
        model.eval()
        test_words = list(u"春江花朝秋月夜")
        for word in test_words:
            # 注意：生成时限制最大长度 = min(Config.max_gen_len, Config.transformer_max_len - 10)
            max_gen = min(Config.max_gen_len, Config.transformer_max_len - 10)
            gen_poetry = ''.join(generate(model, word, ix2word, word2ix,
                                          use_transformer=Config.use_transformer,
                                          max_gen_len=max_gen))
            print(gen_poetry[:200])  # 打印前200字避免刷屏
            f.write(gen_poetry + "\n\n\n")
            f.flush()

        # 保存模型
        save_path = f"{Config.model_prefix}_{epoch+1}.pth"
        t.save(model.state_dict(), save_path)

    f.close()

if __name__ == '__main__':
    train()