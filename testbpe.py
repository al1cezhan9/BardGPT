import json
from bpe import encode, decode

modelID = '[2048V]'

# 1. Load the exact state your train.py will use
with open(f"{modelID}model/vocab.json", "r", encoding="utf-8") as f:
    stoi = json.load(f)
itos = {int(i): s for s, i in stoi.items()} 

with open(f"{modelID}model/merges.txt", "r", encoding="utf-8") as f:
    bpe_merges = f.read().split('\n')[:-1] 
merges_dict = {tuple(pair.split()): i for i, pair in enumerate(bpe_merges)}

# 2. The Test
test_string = "Hello. I really hope that this works!"
print(f"Original: {test_string}")

# Encode
ids = encode(test_string, stoi, merges_dict)
print(f"Encoded IDs: {ids}")

# Decode
reconstructed = decode(ids, itos)
print(f"Reconstructed: {reconstructed}")

# 3. The Guarantee
assert test_string == reconstructed, "Tokenization is lossy! Data was destroyed."
print("\nSuccess: 100% lossless conversion.")

