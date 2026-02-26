import torch
import json
from model import GPT
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

# load the same mappings we created in train.py
with open('vocab.json', 'r', encoding='utf-8') as f:
    meta = json.load(f)
stoi = meta['stoi']
itos = {int(k): v for k, v in meta['itos'].items()}
vocab_size = meta['vocab_size']
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

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