import torch
import json
from model import GPT
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

# load the flat BPE vocabulary
with open('vocab.json', 'r', encoding='utf-8') as f:
    stoi = json.load(f)

itos = {int(i): s for s, i in stoi.items()}
vocab_size = len(stoi)

# load merge rules
with open("merges.txt", "r", encoding="utf-8") as f:
    bpe_merges = f.read().split('\n')[:-1] 
merges_dict = {tuple(pair.split()): i for i, pair in enumerate(bpe_merges)}

from bpe import encode as bpe_encode
from bpe import decode as bpe_decode
# wrappers
encode = lambda s: bpe_encode(s, stoi, merges_dict)
decode = lambda l: bpe_decode(l, itos)

model = GPT(vocab_size)
checkpoint = torch.load('transformer.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval() # set dropout / batchnorm to eval mode

prompt = input("Enter a starting prompt (or press Enter for a random start): ")
if prompt == "":
    # start w newline/null token
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
else:
    context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)

print("\n--- Generating ---\n")
generated_indices = model.generate(context, max_new_tokens=500)
print(decode(generated_indices[0].tolist()))