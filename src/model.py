import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelfAttention(nn.Module):
    def __init__(self, config):
        super(SelfAttention, self).__init__()
        self.config = config
        self.n_head = self.config.num_head
        self.n_embedd = self.config.embdd_dim
        # to be splitted into q ,k, v
        self.c_attn = nn.Linear(self.n_embedd, 3*self.n_embedd)
        self.c_proj = nn.Linear(self.config.embdd_dim, self.config.embdd_dim)
        self.attn_drop = nn.Dropout(self.config.attn_drop)
        self.resdrop = nn.Dropout(self.config.multihead_drop_value)
        self.register_buffer('tril', torch.tril(torch.ones(
            self.config.block_size, self.config.block_size)).view(1, 1, self.config.block_size, self.config.block_size))

    def forward(self, x):
        B, T, C = x.size()  # Batch_size x sequence_length x embeddingdim
        q, k, v = self.c_attn(x).split(self.n_embedd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # Barch_size x n_heads x sequenxe x sequence
        out = (q @ k.transpose(-2, -1)) * (1/math.sqrt(k.size(-1)))
        out = out.masked_fill(self.tril[:, :, :T, :T] == 0, float('-inf'))
        out = F.softmax(out, dim=-1)
        out = self.attn_drop(out)
        out = out @ v  # [32, 20 x 12 x 64]
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resdrop(self.c_proj(out))
        return out


class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.config = config
        self.ln1 = nn.LayerNorm(self.config.embdd_dim)
        self.attn = SelfAttention(self.config)
        self.ln2 = nn.LayerNorm(self.config.embdd_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.config.embdd_dim, self.config.embdd_dim*4),
            nn.Linear(self.config.embdd_dim*4, self.config.embdd_dim),
            nn.GELU(),
            nn.Dropout(self.config.attn_drop))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(
            self.config.vocab_size, self.config.embdd_dim).to(self.config.device)
        self.position_embedding_table = nn.Embedding(
            self.config.block_size, self.config.embdd_dim).to(self.config.device)
        self.blocks = nn.Sequential(*[Block(self.config)
                                      for _ in range(self.config.num_blocks)]).to(self.config.device)
        self.ln_end = nn.LayerNorm(
            self.config.embdd_dim).to(self.config.device)
        self.linear = nn.Linear(self.config.embdd_dim,
                                self.config.vocab_size, bias=False).to(self.config.device)

    def forward(self, x):
        B, T = x.shape
        # Ensure input tensor is on the correct device
        x = x.to(self.config.device)
        token_embdd = self.token_embedding_table(
            x)  # (B,T,C)
        pos_embedd = self.position_embedding_table(torch.arange(
            T, device=self.config.device, dtype=torch.long)).unsqueeze(0)
        x = token_embdd + pos_embedd
        x = self.blocks(x)
        x = self.ln_end(x)
        x = self.linear(x)
        return x

    def generate(self, model, prompt, max_tokens, temp=0.7, top_k=1):
        """
    Generates text by iteratively feeding the model the current prompt, getting
    predictions for the next word, and choosing the next word based on a temperature
    parameter and optionally selecting from the top k most likely words.

    Args:
        model: The language model to use for generation.
        prompt: The initial prompt to start generating text from (torch.Tensor).
        max_tokens: The maximum number of tokens to generate (int).
        temp: The temperature parameter for controlling randomness (float).
        top_k: The number of top words to consider for sampling (int, optional).

    Returns:
        torch.Tensor: The generated text as a tensor.
    """

        for _ in range(max_tokens * 4):
            prompt = prompt[:, -self.config.block_size:].to(self.config.device)
            logits = model(prompt).to(self.config.device)
            logits = logits[:, -1, :]
            logits = logits / temp
            probs = nn.functional.softmax(
                logits, dim=-1).to(self.config.device)
            if top_k > 1:
                values, indices = torch.topk(probs, top_k, dim=-1)
                next_prompt = torch.multinomial(
                    values, num_samples=1).squeeze(1).to(self.config.device)
            else:
                next_prompt = torch.multinomial(
                    probs, num_samples=1).squeeze(1).to(self.config.device)
            next_prompt = next_prompt.unsqueeze(1).to(self.config.device)
            prompt = torch.cat((prompt, next_prompt),
                               dim=1).to(self.config.device)

        return prompt
