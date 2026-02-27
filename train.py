import torch
import os
from model import GPT, block_size
import json
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

# --hyperparams--
batch_size = 64 # independent sequences processed in parallel
max_iters = 10000 # total num iterations of training 
eval_interval = 500 # how often we check loss during training
eval_iters = 200 # how many batches to eval loss on before GD
learning_rate = 3e-4

print(f"Device: {device}")

torch.manual_seed(1337)

# import BPE functions 
from bpe import encode, decode
# load trained BPE vocab
with open("vocab.json", "r", encoding="utf-8") as f:
  stoi = json.load(f)
# load merge rules
with open("merges.txt", "r", encoding="utf-8") as f:
  # read lines, drop trailing empty line
  bpe_merges = f.read().split('\n')[:-1] 
# load dataset (we still need it)
with open('input.txt', 'r', encoding='utf-8') as f:
  text = f.read()

# define reverse mapping + vocab_size
itos = {int(i): s for s, i in stoi.items()} 
vocab_size = len(stoi)

# rebuild the dict: (chunk1, chunk2) -> priority rank
merges_dict = {tuple(pair.split()): i for i, pair in enumerate(bpe_merges)}

# pass the vocabulary + chronological merge rules
data = torch.tensor(encode(text, stoi, merges_dict), dtype=torch.long)

# split data into 90% train vs. 10% validation
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
  # generate small batch of data (input & output)
  data = train_data if split == 'train' else val_data
  # get random position (-block_size) to account for valid starting positions
  # get {batch_size} number of these random offsets
  ix = torch.randint(len(data)-block_size, (batch_size,))
  # x and y just hold the end points
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])
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

m = GPT(vocab_size).to(device)

# time to train!
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate, weight_decay=0.1)

# altered from transformer.pth
checkpoint_path = 'transformer_opt.pth'

if os.path.exists(checkpoint_path):
  print(f"Loading existing weights from {checkpoint_path}...")
  # ensures we load correctly to MPS or CPU
  checkpoint = torch.load(checkpoint_path, map_location=device)
  # restore weights
  m.load_state_dict(checkpoint['model_state_dict'])
  # restore momentum/state 
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  start_iter = checkpoint['iter']
  print("Resuming training from where we left off.")
else:
  start_iter = 0
  print("No checkpoint found. Starting fresh.")

scaler = torch.amp.GradScaler()

for iter in range(start_iter, max_iters):
  if iter % eval_interval == 0:
    losses = estimate_loss()
    print(f"step # {iter} || train loss: {losses['train']:.4f} || val loss: {losses['val']:.4f}")
    checkpoint = {
      'iter': iter,
      'model_state_dict': m.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, 'transformer.pth')
    print(f"Checkpoint at iter {iter} saved!")
  xb, yb = get_batch('train')
  optimizer.zero_grad(set_to_none=True) # zero out grads
  with torch.autocast(device_type=device, dtype=torch.float16):
    logits, loss = m(xb, yb) # mixed precision handling
  scaler.scale(loss).backward() # amplify loss, get grads for each
  scaler.unscale_(optimizer) # unscale grads
  torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0) # clip exploding grads 
  scaler.step(optimizer) # use grad to update params
  scaler.update() 

checkpoint = {
    'iter': iter,
    'model_state_dict': m.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}
torch.save(checkpoint, 'transformer.pth')
print("Model weights saved!")
