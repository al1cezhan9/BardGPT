import torch
import os
from model import GPT, GPTConfig
import json
import math
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
from torch.optim.lr_scheduler import ReduceLROnPlateau
from bpe import encode, decode

# --hyperparams--
batch_size = 64 # independent sequences processed in parallel
max_iters = 2500 # total num iterations of training 
eval_interval = 100 # how often we check loss during training
eval_iters = 100 # how many batches to eval loss on before GD
learning_rate = 3e-4
torch.manual_seed(1337)
print(f"Device: {device}")

modelID = '[1064V]'

# load trained BPE vocab, merge rules, dataset
with open(f"{modelID}model/vocab.json", "r", encoding="utf-8") as f:
  stoi = json.load(f)

# define reverse mapping + vocab_size
itos = {int(i): s for s, i in stoi.items()} 
vocab_size = len(stoi) + 1
print(f"Baseline untrained loss = {math.log(vocab_size)}")

with open(f"{modelID}model/merges.txt", "r", encoding="utf-8") as f:
  bpe_merges = f.read().split('\n')[:-1] # read lines, drop trailing empty line
with open('input.txt', 'r', encoding='utf-8') as f:
  text = f.read()

# rebuild the dict: (chunk1, chunk2) -> priority rank
merges_dict = {tuple(pair.split()): i for i, pair in enumerate(bpe_merges)}
# pass the vocabulary + chronological merge rules
data = torch.tensor(encode(text, stoi, merges_dict), dtype=torch.long)

# split data into 90% train vs. 10% validation
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

config = GPTConfig(vocab_size=vocab_size) 
m = GPT(config).to(device)

def get_batch(split):
  # generate small batch of data (input & output)
  data = train_data if split == 'train' else val_data
  # get random position (-block_size) to account for valid starting positions
  # get {batch_size} number of these random offsets
  ix = torch.randint(len(data)-m.config.block_size, (batch_size,))
  # x and y just hold the end points
  x = torch.stack([data[i:i+m.config.block_size] for i in ix])
  y = torch.stack([data[i+1:i+m.config.block_size+1] for i in ix])
  return x.to(device), y.to(device)

# provides stable estimate of model performance
@torch.no_grad() # tell pytorch we won't call backprop, efficiency 
def estimate_loss():
    out = {} # will hold train/test losses
    m.eval() # put model into eval mode (no dropout, avg batchnorm)
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters) # create tensor to store loss across all batches
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = m(X, Y) # forward pass to get predictions
            losses[k] = loss.item() # get the scalar value
        out[split] = losses.mean() # avg loss for for that split (Train or Test)
    m.train() # back to training mode
    return out

# time to train!
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate, weight_decay=0.1)
# monitor val loss: if it stops improving for N steps, reduce LR
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# altered from transformer.pth
checkpoint_path = f'{modelID}model/transformer.pth'

if os.path.exists(checkpoint_path):
    print(f"Loading existing weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    m.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict']) # restore momentum/state 
    start_iter = checkpoint['metrics']['iters'][-1]+1
    metrics = checkpoint['metrics']
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    print("Resuming training from where we left off.")
else:
    start_iter = 0
    metrics = {'train_loss': [], 'val_loss': [], 'iters': []}
    print("No checkpoint found. Starting fresh.")

scaler = torch.amp.GradScaler()

if metrics['val_loss']: # evaluates to True if list is not empty
    best_val_loss = min(metrics['val_loss'])
else:
    best_val_loss = float('inf')

# conditionally pick best 16-bit format for hardware
if device == 'cuda':
    ptdtype = torch.float16 
elif device == 'mps':
    ptdtype = torch.bfloat16 # Apple Silicon supports Brain Float natively!
else:
    ptdtype = torch.float32 # Fallback for CPU

# 2. Only enable the scaler if we are on CUDA AND using standard float16
scaler_enabled = (device == 'cuda' and ptdtype == torch.float16)
scaler = torch.amp.GradScaler(enabled=scaler_enabled)


try:
    for iter in range(start_iter, max_iters):
        if iter % eval_interval == 0:
            losses = estimate_loss()
            scheduler.step(losses['val'])
            print(f"step # {iter} \n    train loss: {losses['train']:.4f} || val loss: {losses['val']:.4f} || train perp: {math.exp(losses['train']):.4f} || val perp: {math.exp(losses['val']):.4f}")
            metrics['iters'].append(iter)
            metrics['train_loss'].append(losses['train'])
            metrics['val_loss'].append(losses['val'])
            checkpoint = {
                'iter': iter,
                'model_state_dict': m.state_dict(), # dict w/ string names -> raw tensor matrices
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'metrics': metrics,
                'config': {'block_size': m.config.block_size, 'n_embd': m.config.n_embd, 'n_layer': m.config.n_layer, 'n_head': m.config.n_head}
            }
            torch.save(checkpoint, f'{modelID}model/checkpoint_latest.pth')
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                torch.save(checkpoint, f'{modelID}model/checkpoint_best.pth')
                print(f"New best model saved at iter {iter}!")

        xb, yb = get_batch('train', )
        optimizer.zero_grad(set_to_none=True) # zero out grads
        with torch.autocast(device_type=device, dtype=ptdtype):
            logits, loss = m(xb, yb) # mixed precision handling
        scaler.scale(loss).backward() # amplify loss, get grads for each
        scaler.unscale_(optimizer) # unscale grads
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0) # clip exploding grads 
        scaler.step(optimizer) # use grad to update params
        scaler.update() 
        
except KeyboardInterrupt:
    print("\nTraining interrupted by user. Saving current state...")

# This code runs whether the loop finishes OR is interrupted
checkpoint = {
    'iter': iter,
    'model_state_dict': m.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(), 
    'metrics': metrics,
    'config': {'block_size': m.config.block_size, 'n_embd': m.config.n_embd, 'n_layer': m.config.n_layer, 'n_head': m.config.n_head}
}
torch.save(checkpoint, f'{modelID}model/transformer.pth')
print("Model weights saved!")

