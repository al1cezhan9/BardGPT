import torch
import json
from model import GPT, GPTConfig
from torch.nn import functional as F
import time
import os
import matplotlib.pyplot as plt
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

modelID = '[2048V]'

temperature = 1.0
top_k = 10

# load the flat BPE vocabulary
with open(f'{modelID}model/vocab.json', 'r', encoding='utf-8') as f:
    stoi = json.load(f)

itos = {int(i): s for s, i in stoi.items()}
vocab_size = len(stoi) + 1

# load merge rules
with open(f"{modelID}model/merges.txt", "r", encoding="utf-8") as f:
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
checkpoint_path = f"{modelID}model/transformer.pth"
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

# extract saved config dictionary and rebuild the blueprint
saved_args = checkpoint['config']
# unpact the dict directly into the dataclass via **
config = GPTConfig(**saved_args, vocab_size=vocab_size) 

# build the model using the blueprint, then load weights
model = GPT(config) 
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

def dump_generate():
    print('\n=======DUMPING=======')
    global context
    generated_indices = model.generate(context, max_new_tokens=to_generate)
    print(decode(generated_indices[0].tolist()))

def stream_generate():
    for _ in range(to_generate):  
        global context  
        # Crop: don't exceed model's maximum context length
        idx_cond = context[:, -config.block_size:]
        
        # Forward pass: Feed the entire cropped sequence
        # using model() directly to get logits, bypassing model.generate()
        logits, _ = model(idx_cond)
        
        # Pluck: only care about the prediction for the very last timestep
        logits = logits[:, -1, :] / temperature # Shape becomes (B, C)
        
        # Filter: Top-K
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('Inf')
        
        # Sample: Convert to probabilities and pick the next token
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1) # Shape (B, 1)
        
        # Update: Append the new token to running sequence
        context = torch.cat((context, next_token), dim=1)
        
        # Stream: Decode just the single new token and print immediately
        word = decode([next_token.item()])
        print(word, end='', flush=True)


def plot_generation_speed():
    print('\n=======PLOTTING=======')
    global context
    measured_times = []
    steps = []
    model.eval()
    # avoid printing text to avoid confounding factors
    with torch.no_grad():
        for step in range(to_generate):
            start_time = time.perf_counter()
            # crop
            idx_cond = context[:, -config.block_size:]
            # forward pass
            logits, _ = model(idx_cond)
            # pluck + filter
            logits = logits[:, -1, :] / temperature
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            # sample + updated
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            context = torch.cat((context, next_token), dim=1)
            # time it
            end_time = time.perf_counter()
            # convert to millis
            step_ms = (end_time - start_time) * 1000 
            measured_times.append(step_ms)
            steps.append(step + 1)

    plt.plot(steps, measured_times, color='#D0021B', linewidth=1.5, alpha=0.8, label='Empirical Time per Token')
    plt.axvline(x=config.block_size, color='blue', linestyle=':', linewidth=2, 
                label=f'Context Limit ({config.block_size})')

    plt.title("Empirical Time to Generate Each Token", fontsize=14, fontweight='bold')
    plt.xlabel("Generation Step", fontsize=12)
    plt.ylabel("Time (Milliseconds)", fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    save_location = os.path.join(os.getcwd(), 'plots', 'token_vs_gentime.png')
    plt.savefig(save_location)
    plt.show()

stream_generate()
# dump_generate()

# plot_generation_speed()


# # load your trained model
# model.load_state_dict(torch.load("model.pt"))
# model.eval()

# # example input with correct shape
# dummy_input = ???  # change to your input shape

# torch.onnx.export(
#     model,
#     dummy_input,
#     "model.onnx",
#     input_names=["input"],
#     output_names=["output"],
#     dynamic_axes={"input": {0: "batch_size"}}
# )
