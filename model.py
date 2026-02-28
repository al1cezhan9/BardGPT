import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 256 # context length, 1-256 predict 257
    vocab_size: int = 314 # BPE size
    n_layer: int = 4
    n_head: int = 4 
    n_embd: int = 128 # embedding dimensions
    dropout: float = 0.2 # randomly sever connections during training -> mimic ensemble

class Head(nn.Module):
  def __init__(self, config, head_size):
    super().__init__()
    self.key = nn.Linear(config.n_embd, head_size, bias=False)
    self.query = nn.Linear(config.n_embd, head_size, bias=False)
    self.value = nn.Linear(config.n_embd, head_size, bias=False)
    self.dropout = nn.Dropout(config.dropout)
    # tril is not a param of model, it's a buffer
    self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))

  def forward(self, x):
    B,T,C = x.shape
    k = self.key(x)
    q = self.query(x)
    v = self.value(x)
    wei = q @ k.transpose(-2, -1) * (k.shape[-1]**-0.5)
    wei = wei.masked_fill(self.tril[:T, :T]==0, float('-inf'))
    wei = F.softmax(wei, dim=-1)
    wei = self.dropout(wei)
    out = wei @ v
    return out
  
class MultiHeadAttention(nn.Module):
  def __init__(self, config):
    super().__init__()
    head_size = config.n_embd // config.n_head
    self.heads = nn.ModuleList([Head(config, head_size) for _ in range(config.n_head)])
    concatenated_dim = config.n_head * head_size
    self.proj = nn.Linear(concatenated_dim, config.n_embd)
    self.dropout = nn.Dropout(config.dropout)
  
  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    return self.dropout(self.proj(out)) # project layer outcome back onto main highway

# done independently on a token-level basis
class FeedForward(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(config.n_embd, 4 * config.n_embd), # x4dim from OG paper
        nn.ReLU(),
        nn.Linear(4 * config.n_embd, config.n_embd),
        nn.Dropout(config.dropout),
    )
  def forward(self, x):
    return self.net(x)

class Block(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.sa = MultiHeadAttention(config)
    self.ffwd = FeedForward(config)
    self.ln1 = nn.LayerNorm(config.n_embd)
    self.ln2 = nn.LayerNorm(config.n_embd)
  def forward(self, x):
    # residual streams enable clean backprop
    x = x + self.sa(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))
    return x
  
class GPT(nn.Module):
  
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
    self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
    self.blocks = nn.Sequential(
      *[Block(config) for _ in range(config.n_layer)],
    )
    self.ln_f = nn.LayerNorm(config.n_embd)
    # go from token embed -> logits
    self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    # inherit weights from embedding matrix (since they are effectively inverse operations)
    # weight typing for efficiency, 
    self.lm_head.weight = self.token_embedding_table.weight
    self.apply(self._init_weights)

  # init model weights
  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

  def forward(self, idx, targets=None):
    B,T=idx.shape
    # idx and targets are both (B,T) tensor
    tok_embd = self.token_embedding_table(idx) # (B,T,n_embd)
    pos_embd = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T,n_embd)
    x = tok_embd + pos_embd
    x = self.blocks(x)
    x = self.ln_f(x) # final cleanup after stacking residuals
    logits = self.lm_head(x) 

    if targets is None:
        loss = None
    else:
        # python wants (B,C,T) instead
        # stretch out into 2D array
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
    return logits, loss
  
  def generate(self, idx, max_new_tokens, temperature=1.0):
    # idx = (B, T) array of indices rn
    for _ in range(max_new_tokens):
      # crop idx to last block_size token
      idx_cond = idx[:, -self.config.block_size:]
      # get predictions
      logits, loss = self(idx_cond)
      # take only the last time step prediction
      logits = logits[:, -1, :] # becomes (B, C)
      logits = logits / temperature # creativity!
      # apply softmax: logits -> probabilities
      probs = F.softmax(logits, dim=-1) # (B, C)
      # sample from distribution to get next token
      idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
      # append sampled index to running sequence
      idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
    return idx
  