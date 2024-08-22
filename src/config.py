import torch


class Config:
    block_size = 256
    embdd_dim = 480
    num_blocks = 4
    num_head = 12
    head_size = embdd_dim//num_head
    attn_drop = 0.25
    multihead_drop_value = 0.20
    epochs = 1
    vocab_size = 50257
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    lr = 5e-4
    model_name = 'minigpt2'
    wd = 1e-5
    bs = 64
    batch_size = 12
