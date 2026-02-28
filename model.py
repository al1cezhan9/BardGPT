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

def apply_rope(x, start_pos, device):
  # x: (B, T, n_head, head_size)
  B, T, nh, hs = x.shape
  # calc frequencies
  inv_freq = 1.0 / (10000 ** (torch.arange(0, hs, 2).float().to(device) / hs))
  t = torch.arange(start_pos, start_pos + T, device=device).float()
  freqs = torch.outer(t, inv_freq) # (T, hs/2)
  
  # cast to float32 for the math
  cos = freqs.cos().view(1, T, 1, hs//2)
  sin = freqs.sin().view(1, T, 1, hs//2)
  
  # split x into pairs
  x1, x2 = x[..., 0::2], x[..., 1::2]
  # rotate: [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
  out1 = x1 * cos - x2 * sin
  out2 = x1 * sin + x2 * cos
  return torch.stack([out1, out2], dim=-1).flatten(-2)

class Head(nn.Module):
  def __init__(self, config, head_size):
    super().__init__()
    self.head_size = head_size
    self.config = config
    self.key = nn.Linear(config.n_embd, head_size, bias=False)
    self.query = nn.Linear(config.n_embd, head_size, bias=False)
    self.value = nn.Linear(config.n_embd, head_size, bias=False)
    self.dropout = nn.Dropout(config.dropout)
    # tril is not a param of model, it's a buffer
    self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))
    

  # need to handle generation vs. training differently (for KV-caching)
  def forward(self, x, past_kv=None, start_pos=0):
    B,T_query,C = x.shape
    q = self.query(x).view(B, T_query, 1, self.head_size)
    k = self.key(x).view(B, T_query, 1, self.head_size)
    v = self.value(x)
    # handle positions here!
    q = apply_rope(q, start_pos, x.device).view(B, T_query, self.head_size)
    k = apply_rope(k, start_pos, x.device).view(B, T_query, self.head_size)
    # KV-caching for generation
    if past_kv is not None:
      past_k, past_v = past_kv
      k = torch.cat([past_k, k], dim=1)
      v = torch.cat([past_v, v], dim=1)
      # sliding window: evict oldest entries beyond block_size
      if k.shape[1] > self.config.block_size:
          k = k[:, -self.config.block_size:]
          v = v[:, -self.config.block_size:]

    present_kv = (k, v)
    T_key = k.shape[1]
    wei = q @ k.transpose(-2, -1) * (self.head_size**-0.5)
    # If > 1, processing multiple tokens at once (Training) -> MUST MASK
    # If == 1, predicting exactly one token (Generation) -> NO MASK
    if T_query > 1:
        mask = self.tril[:T_query, :T_key] 
        wei = wei.masked_fill(mask == 0, float('-inf'))
    wei = F.softmax(wei, dim=-1)
    wei = self.dropout(wei)
    out = wei @ v
    return out, present_kv
  
class MultiHeadAttention(nn.Module):
  def __init__(self, config):
    super().__init__()
    head_size = config.n_embd // config.n_head
    self.heads = nn.ModuleList([Head(config, head_size) for _ in range(config.n_head)])
    concatenated_dim = config.n_head * head_size
    self.proj = nn.Linear(concatenated_dim, config.n_embd)
    self.dropout = nn.Dropout(config.dropout)
  
  def forward(self, x, past_kvs=None, start_pos=0):
    if past_kvs is None:
      past_kvs = [None] * len(self.heads)
    outs = []
    present_kvs = []
    # look through all heads (have learned different K's and V's)
    for i, head in enumerate(self.heads):
      # give Head[i] its specific past_kv[i]
      out, present_kv = head(x, past_kv=past_kvs[i], start_pos=start_pos)
      outs.append(out)               # Collect outputs
      present_kvs.append(present_kv) # Collect new memory tuples

    out = torch.cat(outs, dim=-1)
    # project layer outcome back onto main highway
    return self.dropout(self.proj(out)), present_kvs 

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
  def forward(self, x, past_kvs=None, start_pos=0):
    sa_out, present_kvs = self.sa(self.ln1(x), past_kvs=past_kvs, start_pos=start_pos)

    # residual streams enable clean backprop
    x = x + sa_out
    x = x + self.ffwd(self.ln2(x))
    return x, present_kvs
  
class GPT(nn.Module):
  
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
    # self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
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
  
  def forward(self, idx, targets=None, past_kvs=None, position_offset=0):
    B,T_query=idx.shape
    # adjust position for KV-caching
    # past_length = 0 if past_kvs is None else past_kvs[0][0][0].shape[1]
    # start_pos = past_length
    start_pos = position_offset
    
    x = self.token_embedding_table(idx)

    working_kvs = past_kvs if past_kvs is not None else [None] * len(self.blocks)
    present_kvs = []
    for i, block in enumerate(self.blocks):
        # pass the specific block its specific past memory
        x, pk = block(x, past_kvs=working_kvs[i], start_pos=start_pos)
        present_kvs.append(pk)

    x = self.ln_f(x) # final cleanup after stacking residuals
    logits = self.lm_head(x) 

    if targets is None:
        loss = None
    else:
        # python wants (B,C,T) instead
        # stretch out into 2D array
        B, T_out, C = logits.shape
        loss = F.cross_entropy(logits.view(B*T_out, C), targets.view(B*T_out))
    return logits, loss, present_kvs
  
  @torch.no_grad()
  def generate(self, idx, max_new_tokens, temperature=1.0):
    self.eval()
    past_kvs = None
    # idx = (B, T) array of indices rn
    for _ in range(max_new_tokens):
      if past_kvs is None:
          # no memory yet... pass whole sequence to build/populate the cache
          idx_input = idx[:, -self.config.block_size:]
      else:
          # memory! Only pass the single newest token
          idx_input = idx[:, -1:]

      logits, loss, present_kvs = self(idx_input, past_kvs=past_kvs)
      past_kvs = present_kvs

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
  