import torch
import os
from model import GPT, block_size
import json
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

# --hyperparams--
batch_size = 64 # independent sequences processed in parallel
max_iters = 5000 # total num iterations of training 
eval_interval = 500 # how often we check loss during training
eval_iters = 200 # how many batches to eval loss on before GD
learning_rate = 3e-4

print(f"Device: {device}")

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# get the vocab by sorting the set of the input text (str)
chars = sorted(list(set(text)))
vocab_size = len(chars)

# map char (str) to integer by index in the chars list
# tradeoff between size of vocab and dimension of embedding
ctoi = {ch:i for i,ch in enumerate(chars)}
itoc = {i:ch for i,ch in enumerate(chars)}

# save these functions as metadata to load on inference
meta = {
    'vocab_size': vocab_size,
    'itos': itoc,
    'stoi': ctoi
}
with open('vocab.json', 'w', encoding='utf-8') as f:
  json.dump(meta, f)

print(f"Vocab of size {vocab_size} saved to vocab.json")

# lambda = mini-functions
encode = lambda s: [ctoi[c] for c in s]
decode = lambda l: ''.join([itoc[i] for i in l])

# encode entire dataset
# store it into a torch.Tensor (multi-d array optimized for computation)
data = torch.tensor(encode(text), dtype=torch.long)
# print(data.shape, data.dtype)
# print(data[:100])

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

model = GPT(vocab_size)
m = model.to(device)

# # 1x1 tensor holding a 0, kickoff character
# idx = torch.zeros((1, 1), dtype=torch.long)
# print("="*20+"before training: "+"="*20)
# print(decode(m.generate(idx, max_new_tokens=100)[0].tolist()))

# time to train!
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate, weight_decay=0.1)

checkpoint_path = 'transformer.pth'

if os.path.exists(checkpoint_path):
  print(f"Loading existing weights from {checkpoint_path}...")
  # ensures we load correctly to MPS or CPU
  checkpoint = torch.load(checkpoint_path, map_location=device)
  # restore weights
  m.load_state_dict(checkpoint['model_state_dict'])
  # restore momentum/state 
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  print("Resuming training from where we left off.")
else:
  print("No checkpoint found. Starting fresh.")


for iter in range(max_iters):
  if iter % eval_interval == 0:
    losses = estimate_loss()
    print(f"step # {iter} || train loss: {losses['train']:.4f} || val loss: {losses['val']:.4f}")
  xb, yb = get_batch('train')
  logits, loss = m(xb, yb)
  optimizer.zero_grad(set_to_none=True) # zero out gradients
  loss.backward() # get grads for all params
  optimizer.step() # using grad to update params

checkpoint = {
    'model_state_dict': m.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}
torch.save(checkpoint, 'transformer.pth')
print("Model weights saved!")
