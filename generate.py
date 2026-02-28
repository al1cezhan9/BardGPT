import torch
import json
from model import GPT, GPTConfig
from torch.nn import functional as F
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

prompt = input("Enter a starting prompt (or press Enter for a random start): ")
if prompt == "":
    # start w newline/null token
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
else:
    context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)

to_generate = input("Enter the number of tokens to generate (or press Enter for default): ")
if to_generate == "":
    to_generate = 500
else: 
    to_generate = int(to_generate)

# load checkpoint
checkpoint_path = 'transformer.pth'
checkpoint = torch.load(checkpoint_path, map_location=device)

# extract saved config dictionary and rebuild the blueprint
saved_args = checkpoint['config']
# unpact the dict directly into the dataclass via **
config = GPTConfig(**saved_args, vocab_size=vocab_size) 

# build the model using the blueprint, then load weights
model = GPT(config) 
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

print("\n--- Generating ---\n")
#### texts appears all at once
# generated_indices = model.generate(context, max_new_tokens=to_generate)
# print(decode(generated_indices[0].tolist()))

past_kvs = None # init empty KV cache
temperature = 1.0
# instead of using model.generate(), pull that logic out for all us impatient souls
# for _ in range(to_generate):    
#     # If we have memory, only pass the last token
#     idx_input = context[:, -1:] if past_kvs is not None else context
    
#     logits, _, present_kvs = model(idx_input, past_kvs=past_kvs)
#     past_kvs = present_kvs # Update memory
    
#     # sample next token
#     logits = logits[:, -1, :] / temperature
#     # top-k filtering
#     v, _ = torch.topk(logits, min(10, logits.size(-1)))
#     logits[logits < v[:, [-1]]] = -float('Inf')
    
#     probs = F.softmax(logits, dim=-1)
#     next_token = torch.multinomial(probs, num_samples=1)
    
#     # update the running sequence
#     context = torch.cat((context, next_token), dim=1)
    
#     # decode just the single new token
#     token_id = next_token.item()
#     word = decode([token_id])
    
#     # print w/o newline and force terminal to show immediately
#     print(word, end='', flush=True)


past_kvs = None
total_generated = 0

for _ in range(to_generate):
    if past_kvs is None:
        idx_input = context
        position_offset = 0
    else:
        idx_input = context[:, -1:]
        position_offset = total_generated + context.shape[1] - 1

    logits, _, present_kvs = model(idx_input, past_kvs=past_kvs, position_offset=position_offset)
    past_kvs = present_kvs
    total_generated += 1

    # sample next token
    logits = logits[:, -1, :] / temperature

    # top-k filtering
    v, _ = torch.topk(logits, min(10, logits.size(-1)))
    logits[logits < v[:, [-1]]] = -float('Inf')
    
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    
    # update the running sequence
    context = torch.cat((context, next_token), dim=1)
    
    # decode just the single new token
    token_id = next_token.item()
    word = decode([token_id])
    
    # print w/o newline and force terminal to show immediately
    print(word, end='', flush=True)


# @torch.no_grad()
# def debug_generate(model, idx, max_new_tokens):
#     model.eval()
#     past_kvs = None
    
#     for _ in range(max_new_tokens):
#         # We need to capture the attention weights, so we'll modify the forward pass slightly
#         # For a quick debug, let's just look at the last token's attention in the first head
#         idx_input = idx[:, -1:] if past_kvs is not None else idx
        
#         # We manually call the layers to peek at the 'wei' (attention scores)
#         # Note: This is a simplified trace for the FIRST block's FIRST head
#         logits, _, present_kvs = model(idx_input, past_kvs=past_kvs)
#         past_kvs = present_kvs
        
#         # --- DEBUG VIZ ---
#         # If you want to see if the model is "confused", check the logit distribution
#         probs = F.softmax(logits[:, -1, :] / 1.0, dim=-1)
#         top_probs, top_indices = torch.topk(probs, 5)
        
#         print(f"\nNext Token Predictions:")
#         for i in range(5):
#             print(f"  {decode([top_indices[0,i].item()]):<10} : {top_probs[0,i].item():.4f}")
#         # -----------------

#         next_token = torch.multinomial(probs, num_samples=1)
#         idx = torch.cat((idx, next_token), dim=1)
#         print(f"CHOSEN: [{decode([next_token.item()])}]")
        
#         if next_token.item() == stoi.get('<|endoftext|>', -1): break

# # Run it
# debug_generate(model, context, 10)