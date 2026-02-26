import torch
import torch.nn as nn
from torch.nn import functional as F

# --hyperparams--
block_size = 256 # context length, 1-256 predict 257
n_embd = 128 # embedding dimensions
n_layer = 4
n_head = 4
# randomly sever connections during training -> mimic ensemble
dropout = 0.2 

class Head(nn.Module):
  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(n_embd, head_size, bias=False)
    self.query = nn.Linear(n_embd, head_size, bias=False)
    self.value = nn.Linear(n_embd, head_size, bias=False)
    self.dropout = nn.Dropout(dropout)
    # tril is not a param of model, it's a buffer
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
  def forward(self, x):
    B,T,C = x.shape
    k = self.key(x)
    q = self.query(x)
    v = self.value(x)
    wei = q @ k.transpose(-2, -1) * C**-0.5
    wei = wei.masked_fill(self.tril[:T, :T]==0, float('-inf'))
    wei = F.softmax(wei, dim=-1)
    wei = self.dropout(wei)
    out = wei @ v
    return out
  
class MultiHeadAttention(nn.Module):
  def __init__(self, n_head, head_size):
    super().__init__()
    head_size = n_embd // n_head
    self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
    self.proj = nn.Linear(n_embd, n_embd)
    self.dropout = nn.Dropout(dropout)
  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    return self.dropout(self.proj(out)) # project layer outcome back onto main highway

# done independently on a token-level basis
class FeedForward(nn.Module):
  def __init__(self, n_embd):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(n_embd, 4 * n_embd), # x4dim from OG paper
        nn.ReLU(),
        nn.Linear(4 * n_embd, n_embd),
        nn.Dropout(dropout),
    )
  def forward(self, x):
    return self.net(x)

class Block(nn.Module):
  def __init__(self, n_embd, n_head):
    super().__init__()
    head_size = n_embd // n_head
    self.sa = MultiHeadAttention(n_head, head_size)
    self.ffwd = FeedForward(n_embd)
    self.ln1 = nn.LayerNorm(n_embd)
    self.ln2 = nn.LayerNorm(n_embd)
  def forward(self, x):
    # residual streams enable clean backprop
    x = x + self.sa(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))
    return x
  
class GPT(nn.Module):
  
  def __init__(self, vocab_size):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.position_embedding_table = nn.Embedding(block_size, n_embd)
    self.blocks = nn.Sequential(
      *[Block(n_embd, n_head=n_head) for _ in range(n_layer)],
    )
    self.ln_f = nn.LayerNorm(n_embd)
    # go from token embed -> logits
    self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
    # inherit weights from embedding matrix (since they are effectively inverse operations)
    # weight typing for efficiency, 
    self.lm_head.weight = self.token_embedding_table.weight

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
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
        loss = F.cross_entropy(logits, targets)
    return logits, loss
  
  def generate(self, idx, max_new_tokens, temperature=1.0):
    # idx = (B, T) array of indices rn
    for _ in range(max_new_tokens):
      # crop idx to last block_size token
      idx_cond = idx[:, -block_size:]
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
  