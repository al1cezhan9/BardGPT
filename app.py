import streamlit as st
import torch
import json
from model import GPT

# -- Page Config --
st.set_page_config(page_title="Shakespeare GPT", page_icon="")
st.title("Mini-GPT Shakespeare")
st.markdown("A character-level transformer playground.")

# -- Load Assets (Cached) --
@st.cache_resource
def load_model():
    with open('vocab.json', 'r') as f:
        meta = json.load(f)
    
    # Setup mapping
    stoi = meta['stoi']
    itos = {int(k): v for k, v in meta['itos'].items()}
    encode = lambda s: [stoi[c] for c in s if c in stoi]
    decode = lambda l: ''.join([itos[i] for i in l])
    
    # Reconstruct Model
    model = GPT(meta['vocab_size'])
    checkpoint = torch.load('transformer.pth', map_location='cpu')
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