# tokenizer

A byte pair encoding (BPE) tokenizer I wrote from scratch to understand how language models actually turn text into numbers. No tokenizer library, everything starts from raw UTF-8 bytes.

It all lives in one file, `main.py`, and comes in three versions that build on each other.

`BasicTokenizer` is plain byte-level BPE. It turns text into UTF-8 bytes and keeps merging whichever adjacent pair is most frequent. There is no regex splitting, so merges can happen across anything.

`RegexTokenizer` splits the text with the GPT-4 regex first and only runs BPE inside each piece, so a merge never crosses a piece boundary. This is also where special tokens and save/load come in.

`GPT4Tokenizer` doesn't train at all. It reads the merge rules back out of tiktoken's `cl100k_base` and reuses them, which is what lets it match the real GPT-4 tokenizer.

## Install

```bash
pip install -r requirements.txt
```

## Usage

```python
from main import RegexTokenizer

tok = RegexTokenizer()
tok.train(open("taylorswift.txt", encoding="utf-8").read(), vocab_size=512)
tok.register_special_tokens({"<|endoftext|>": 512})

ids = tok.encode("hello<|endoftext|>world", allowed_special="all")
print(ids)
print(tok.decode(ids))          # hello<|endoftext|>world

tok.save("model/taylor")        # writes taylor.model and taylor.vocab
```

Reproducing the GPT-4 split:

```python
from main import GPT4Tokenizer
import tiktoken

tok = GPT4Tokenizer()
enc = tiktoken.get_encoding("cl100k_base")

text = open("taylorswift.txt", encoding="utf-8").read()
print(tok.encode(text) == enc.encode(text))     # True
```

## Verification

I ran `GPT4Tokenizer` against tiktoken on the whole `taylorswift.txt` file, all 185,561 characters, and the two agree token for token (49,298 tokens each). It also holds up on the annoying cases: emoji, joined flag sequences, Arabic and Hindi, source code, URLs. And every tokenizer round-trips, so `decode(encode(x))` gives you back exactly `x`.
