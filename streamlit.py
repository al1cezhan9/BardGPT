import streamlit as st
import torch
import json
from model import GPT, GPTConfig
from generate import bpe_encode, bpe_decode

modelID = '[2048V]'

# -- Page Config --
st.set_page_config(page_title="Shakespeare GPT", page_icon="")
st.title("Mini-GPT Shakespeare")
st.markdown("A character-level transformer playground.")

# -- Load Assets (Cached) --
@st.cache_resource
def load_model():
    # 1. Load the flat BPE vocabulary
    with open(f'{modelID}model/vocab.json', 'r', encoding='utf-8') as f:
        stoi = json.load(f)
    itos = {int(i): s for s, i in stoi.items()}
    vocab_size = len(stoi)

    # 2. Load the BPE merge rules
    with open(f"{modelID}model/merges.txt", "r", encoding="utf-8") as f:
        bpe_merges = f.read().split('\n')[:-1] 
    merges_dict = {tuple(pair.split()): i for i, pair in enumerate(bpe_merges)}

    # 3. Setup BPE Wrappers
    encode = lambda s: bpe_encode(s, stoi, merges_dict)
    decode = lambda l: bpe_decode(l, itos)
    
    # 4. Reconstruct Model using the new GPTConfig
    checkpoint = torch.load(f'{modelID}model/transformer.pth', map_location='cpu', weights_only=False)
    saved_args = checkpoint['config']
    config = GPTConfig(**saved_args, vocab_size=vocab_size) 
    
    model = GPT(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, encode, decode

model, encode, decode = load_model()

# -- Sidebar Controls --
st.sidebar.header("Model Params")
length = st.sidebar.slider("Tokens to generate", 50, 10000, 1000)
temp = st.sidebar.slider("Temperature", 0.1, 2.0, 0.8)

# -- Main Interface --
prompt = st.text_input("Enter a prompt:", value="ROMEO: ")

if st.button("Go!"):
    with st.spinner("Consulting the Bard..."):
        # Convert prompt to tensor
        context = torch.tensor([encode(prompt)], dtype=torch.long)
        
        # Generate
        output_indices = model.generate(context, max_new_tokens=length) 
        output_text = decode(output_indices[0].tolist())
        
        st.subheader("Output:")
        st.code(output_text, language="text")