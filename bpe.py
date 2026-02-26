import regex as re
import json
import collections

# REGEX split to keep words separate
GPT_REGEX = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

def get_stats(vocab):
    """count freq of adjacent pairs across all word chunks"""
    # (chunk1, chunk2) -> freq
    pairs = collections.defaultdict(int) # defaults to 0
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i], symbols[i+1]] += freq
    return pairs

# pair = tuple of two tokens we want to merge
# dict of tok sequences with val = frequencies
def merge_vocab(pair, v_in):
    """merge most frequent pair in all word chunks"""
    v_out = {}
    bigram = re.escape(' '.join(pair)) # treat special regex chars as literal
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)') # only match when separate (not alr smooshed)
    for word in v_in:
        w_out = p.sub(''.join(pair), word) # smoosh the toks tgt for all words that contain it
        v_out[w_out] = v_in[word] # ensure v_out count match v_in
    return v_out


# only train tokenizer if called explicitly
if __name__ == '__main__':
    # load / prep data
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    # chop text into chunks + format as space-separated characters
    # "Hello world" -> {"H e l l o": 1, " w o r l d": 1}
    words = re.findall(GPT_REGEX, text) # slice into words (preserve space)
    vocab = collections.defaultdict(int)
    for word in words:
        # swap invisible characters for visible ones
        # space = Ġ; newline = Ċ; enables model to learn formatting
        safe_word = word.replace(' ', 'Ġ').replace('\n', 'Ċ')
        # space-separated str -> count
        vocab[' '.join(list(safe_word))] += 1

    # BPE learning loop
    num_merges = 250 # target vocabulary size increase (tune?)
    # (chunk1, chunk2) -> priority
    merges = {}

    print("Learning BPE merges...")
    for i in range(num_merges):
        # (chunk1, chunk2) -> count
        pairs = get_stats(vocab) # frequency dictionary of chunk pairs
        if not pairs:
            break # no more pairs to merge
        # find most frequent pair
        # best is a tuple
        best = max(pairs, key=pairs.get)
        # apply merge to our working vocabulary
        vocab = merge_vocab(best, vocab)
        merges[best] = i
        print(f"Merge {i}: {best} -> {best[0]}{best[1]}")
    # map merged strs -> ints; save to vocab.json/train_data.pt
    # finalize the vocab
    subwords = set()
    for word in vocab:
        for token in word.split(): # split on spaces to get subword toks
            subwords.add(token)
    # sort (for reproducibility)
    subwords = sorted(list(subwords))
    # assign int IDs
    stoi = {s: i for i, s in enumerate(subwords)} # chunk -> ID
    itos = {i: s for s, i in stoi.items()}        # ID -> chunk
    # save
    with open("vocab.json", "w", encoding="utf-8") as f:
        json.dump(stoi, f, ensure_ascii=False, indent=2)
    with open("merges.txt", "w", encoding="utf-8") as f:
        for pair in merges.keys():
            f.write(f"{pair[0]} {pair[1]}\n")
    print("BPE Training Complete. Vocabulary densified.")
    print(f"Final Vocab Size: {len(subwords)}") # should be initial alphabet + num merges
    print(f"Our Chunks: {subwords}")

# to be used in train.py
def decode(ids, itos): # trivial
    """translate list of int IDs -> single string"""
    # look up each ID, get the str, and join 
    tokens = [itos[i] for i in ids]
    text = "".join(tokens)
    return text.replace('Ġ', ' ').replace('Ċ', '\n')

def get_pairs(word_symbols):
    """return set of all adjacent pairs in a list of symbols"""
    return set(zip(word_symbols[:-1], word_symbols[1:]))

def encode(text, stoi, merges): # nontrivial
    """translates string -> list of BPE int IDs"""
    # chop the text into chunks the exact training regex
    words = re.findall(GPT_REGEX, text)
    ids = []
    
    for word in words:
        # invisible chars
        safe_word = word.replace(' ', 'Ġ').replace('\n', 'Ċ')
        # start with isolated characters
        symbols = list(safe_word)
        # iteratively apply merges to preserve priority
        while len(symbols) >= 2:
            pairs = get_pairs(symbols)
            # find pair merged EARLIEST during training
            # merges maps pair -> training rank (0 to num_merges)
            best_pair = min(pairs, key=lambda p: merges.get(p, float('inf')))
            # if best pair isn't in our learned merges, we stop
            if best_pair not in merges:
                break
            # smoosh chosen pair together in the symbols list
            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and (symbols[i], symbols[i+1]) == best_pair:
                    new_symbols.append(symbols[i] + symbols[i+1])
                    i += 2 # skip the next symbol since it was merged
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
        # convert final merged chunks -> int IDs
        for symbol in symbols:
            ids.append(stoi[symbol])
    return ids