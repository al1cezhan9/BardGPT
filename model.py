import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 256 # context length, 1-256 predict 257
    vocab_size: int = 2048 # BPE size
    n_layer: int = 6
    n_head: int = 4 
    n_embd: int = 256 # embedding dimensions
    dropout: float = 0.3 # randomly sever connections during training -> mimic ensemble

class Head(nn.Module):
  def __init__(self, config, head_size):
    super().__init__()
    self.key = nn.Linear(config.n_embd, head_size, bias=False)
    self.query = nn.Linear(config.n_embd, head_size, bias=False)
    self.value = nn.Linear(config.n_embd, head_size, bias=False)
    self.dropout = nn.Dropout(config.dropout)
    # tril is not a param of model, it's a buffer
    self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))

  def forward(self, x, return_attn=False):
    B,T,C = x.shape
    k = self.key(x)
    q = self.query(x)
    v = self.value(x)
    wei = q @ k.transpose(-2, -1) * (k.shape[-1]**-0.5)
    wei = wei.masked_fill(self.tril[:T, :T]==0, float('-inf'))
    wei = F.softmax(wei, dim=-1)
    wei = self.dropout(wei)
    out = wei @ v
    attn_weights = wei if return_attn else None
    return out, attn_weights
  
class MultiHeadAttention(nn.Module):
  def __init__(self, config):
    super().__init__()
    head_size = config.n_embd // config.n_head
    self.heads = nn.ModuleList([Head(config, head_size) for _ in range(config.n_head)])
    concatenated_dim = config.n_head * head_size
    self.proj = nn.Linear(concatenated_dim, config.n_embd)
    self.dropout = nn.Dropout(config.dropout)
  
  def forward(self, x, return_attn=False):
    head_outputs = [h(x, return_attn=return_attn) for h in self.heads]
    out = torch.cat([res[0] for res in head_outputs], dim=-1)
    attns = None
    if return_attn:
        # Stack weights: (num_heads, B, T, T)
        attns = torch.stack([res[1] for res in head_outputs]) 
    # project layer outcome back onto main highway
    return self.dropout(self.proj(out)), attns

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
  def forward(self, x, return_attn=False):
    # residual streams enable clean backprop
    sa_out, attns = self.sa(self.ln1(x), return_attn=return_attn)
    x = x + sa_out
    x = x + self.ffwd(self.ln2(x))
    return x, attns
  
class GPT(nn.Module):
  
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
    self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
    self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
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

  def forward(self, idx, targets=None, return_attn=False):
    B,T=idx.shape
    # idx and targets are both (B,T) tensor
    tok_embd = self.token_embedding_table(idx) # (B,T,n_embd)
    pos_embd = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T,n_embd)
    x = tok_embd + pos_embd

    last_attns = None
    for i, block in enumerate(self.blocks):
        # If it's the last block and we want weights, capture them
        is_last = (i == len(self.blocks) - 1)
        x, attns = block(x, return_attn=(return_attn and is_last))
        if is_last:
            last_attns = attns

    x = self.ln_f(x) # final cleanup after stacking residuals
    logits = self.lm_head(x) 

    loss = None
    if targets is not None:
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
    return (logits, loss, last_attns) if return_attn else (logits, loss)
  
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

  def generate_stream(self, idx, max_new_tokens, temperature=1.0):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -self.config.block_size:]
        
        # Call forward with return_attn=True
        # Returns: (logits, loss, last_attns)
        logits, _, attn = self.forward(idx_cond, return_attn=True)
        
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        
        # attn shape is (n_head, B, T, T)
        # We want the last token's attention: (n_head, all_prev_tokens)
        token_attn = attn[:, 0, -1, :].tolist() 
        
        yield idx_next.item(), token_attn
        
        idx = torch.cat((idx, idx_next), dim=1)