import torch
import json
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from model import GPT, GPTConfig
from bpe import encode as bpe_encode
from bpe import decode as bpe_decode
from fastapi.responses import RedirectResponse
from fastapi.responses import FileResponse

modelID = '[2048V]'

def load_assets():
    with open(f'{modelID}model/vocab.json', 'r', encoding='utf-8') as f:
        stoi = json.load(f)
    itos = {int(i): s for s, i in stoi.items()}
    vocab_size = len(stoi) + 1
    
    with open(f"{modelID}model/merges.txt", "r", encoding="utf-8") as f:
        bpe_merges = f.read().split('\n')[:-1] 
    merges_dict = {tuple(pair.split()): i for i, pair in enumerate(bpe_merges)}

    _encode = lambda s: bpe_encode(s, stoi, merges_dict)
    _decode = lambda l: bpe_decode(l, itos)
    
    checkpoint = torch.load(f'{modelID}model/transformer.pth', map_location='cpu', weights_only=False)
    config = GPTConfig(**checkpoint['config'], vocab_size=vocab_size) 
    _model = GPT(config)
    _model.load_state_dict(checkpoint['model_state_dict'])
    _model.eval()
    
    return _model, _encode, _decode, config

model, encode, decode, config = load_assets()

app = FastAPI()

# CORS MIDDLEWARE so HTML can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def read_index():
    return FileResponse('static/index.html')


@app.get("/generate")
async def generate(prompt: str = "ROMEO: ", length: int = 100, temp: float = 0.8):
    def stream_generator():
        if not prompt or prompt.strip() == "":
            tokens = [204] # can tinker with the default start token by checking out the covab from bpe.py
        else:
            tokens = encode(prompt)
        context = torch.tensor([tokens], dtype=torch.long) #

        # Send the prompt tokens to the UI so indices match
        for i, t_id in enumerate(tokens):
            yield json.dumps({
                "text": decode([t_id]),
                "index": i,
                "attn": [[] for _ in range(config.n_head)] # No attn for prompt starts
            }) + "\n"

        # Generate and send new tokens
        # Current length is len(tokens)
        for token_id, attn in model.generate_stream(context, length, temp):
            data = {
                "text": decode([token_id]),
                "attn": attn
            }
            yield json.dumps(data) + "\n"
            context = torch.cat((context, torch.tensor([[token_id]])), dim=1) #

    return StreamingResponse(stream_generator(), media_type="application/x-ndjson")