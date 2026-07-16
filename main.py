import regex as re
import unicodedata



class Tokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {i : bytes([i]) for i in range(256)}
        self.special_tokens = {}
        self.inverse_special_tokens = {}
        self.pattern = ""
    def register_special_tokens(self,special_tokens):
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v:k for k,v in special_tokens.items()}

    def decode(self, ids):
        part_bytes = []
        for idx in ids:
            if idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            elif idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            else:
                raise ValueError(f"invalid token id: {idx}")
        text_bytes = b"".join(part_bytes)
        return text_bytes.decode("utf-8", errors="replace")
    
    def _get_stats(self, ids, counts=None):
        if counts is None:
            counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts
    
    def _merge(self, ids, pair, idx):
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids
    def save(self, file_prefix):

        model_file = file_prefix + ".model"
        with open(model_file, "w", encoding="utf-8") as f:
            f.write("bpe v1\n")
            f.write(f"{self.pattern}\n")
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            for parent1, parent2 in self.merges.keys():
                f.write(f"{parent1} {parent2}\n")
        
        vocab_file = file_prefix + ".vocab"
        inverted_merges = {v: k for k, v in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token_bytes in self.vocab.items():
                s = token_bytes.decode('utf-8', errors='replace')
                s = "".join(ch if not unicodedata.category(ch).startswith("C") else f"\\u{ord(ch):04x}" for ch in s)
                if idx in inverted_merges:
                    p0, p1 = inverted_merges[idx]
                    f.write(f"[{p0}][{p1}] -> [{idx}] {s}\n")
                else:
                    f.write(f"[{idx}] {s}\n")
    def load(self,model_file):
        assert model_file.endswith(".model")
        merges = {}
        special_tokens = {}
        
        
        with open(model_file, "r", encoding="utf-8") as f:
            header = f.readline().strip()
            assert header == "bpe v1"
            self.pattern = f.readline().strip()
            self.compiled_pattern = re.compile(self.pattern)
            num_special = int(f.readline().strip())
            
            for _ in range(num_special):
                special, idx = f.readline().strip().split()
                special_tokens[special] = int(idx)
            
            idx = 256
            for line in f:
                line = line.strip()
                if not line:
                    continue
                p0, p1 = map(int, line.split())
                merges[(p0, p1)] = idx
                idx += 1
                
        self.merges = merges
        self.register_special_tokens(special_tokens)
        
        self.vocab = {i: bytes([i]) for i in range(256)}
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]
        for special, idx in self.special_tokens.items():
            self.vocab[idx] = special.encode("utf-8")
        

class BasicTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)

        for i in range(num_merges):
            stats = self._get_stats(ids)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            new_idx = 256 + i
            self.merges[pair] = new_idx
            self.vocab[new_idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

            ids = self._merge(ids, pair, new_idx)
            
            if verbose:
                print(f"Merged {pair} into {new_idx}, corresponding value : {self.vocab[new_idx]}")
            

    def encode(self, text):
        ids = list(bytes(text.encode("utf-8")))
        while len(ids) >= 2:
            stats = self._get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            new_idx = self.merges[pair]
            ids = self._merge(ids, pair, new_idx)
        return ids

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class RegexTokenizer(Tokenizer):
    def __init__(self,pattern=None):
        super().__init__()
        self.pattern = pattern if pattern else GPT4_SPLIT_PATTERN
        self.compiled_pattern = re.compile(self.pattern)
        

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        chunks = self.compiled_pattern.findall(text)
        ids_list = [list(chunk.encode("utf-8")) for chunk in chunks] 
        
        for i in range(num_merges):
            stats = {}
            for ids in ids_list:
                self._get_stats(ids, stats)
            
            if not stats:
                break

            pair = max(stats, key=stats.get)
            new_idx = 256 + i

            self.merges[pair] = new_idx 
            self.vocab[new_idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

            ids_list = [self._merge(ids, pair, new_idx) for ids in ids_list]

            if verbose:
                print(f"Merged {pair} into {new_idx}, corresponding value: {self.vocab[new_idx]}")

    def _encode_chunk(self, ids):
        while len(ids) >= 2:
            stats = {}
            self._get_stats(ids, stats)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            new_idx = self.merges[pair]
            ids = self._merge(ids, pair, new_idx)
        return ids

    def encode_ordinary(self, text):
        chunks = self.compiled_pattern.findall(text)
        all_ids = []
        for chunk in chunks:
            chunk_bytes = list(chunk.encode("utf-8"))
            chunk_ids = self._encode_chunk(chunk_bytes)
            all_ids.extend(chunk_ids)
        return all_ids
    
    def encode(self, text, allowed_special="none"):
        if allowed_special == "all":
            special_set = set(self.special_tokens.keys())
        elif allowed_special == "none":
            special_set = set()
        elif isinstance(allowed_special, (set, list)):
            special_set = set(allowed_special)
        else:
            raise ValueError(f"unknown allowed_special {allowed_special}")
        
        if not special_set:
            return self.encode_ordinary(text)
        
        special_pattern = re.compile("(" + "|".join(re.escape(t) for t in special_set) + ")")
        parts = special_pattern.split(text)
        ids = []

        for part in parts:
            if part in special_set:
                ids.append(self.special_tokens[part])
            else:
                ids.extend(self.encode_ordinary(part))
        return ids

class GPT4Tokenizer(RegexTokenizer):
    def __init__(self):
        super().__init__()
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")

        mergable_ranks = enc._mergeable_ranks

        self.byte_shuffle = {i: mergable_ranks[bytes([i])] for i in range(256)}
        self.inverse_byte_shuffle = {v : k for k,v in self.byte_shuffle.items()}

        print("recovering merges,may take time...")
        self.merges = self._recover_merges(mergable_ranks)
        print("Success !")

        self.vocab = {idx : bytes([i]) for i,idx in self.byte_shuffle.items()}

        for (p0,p1) , idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]

        self.register_special_tokens(enc._special_tokens)

    def _encode_chunk(self, ids):
        ids = [self.byte_shuffle[b] for b in ids]
        return super()._encode_chunk(ids)

    def _recover_merges(self,mergable_ranks):
        merges = {}

        sorted_ranks = sorted(
            [(rank,b) for b,rank in mergable_ranks.items() if len(b) > 1]
        )

        for rank,b in sorted_ranks:
            ids = [self.byte_shuffle[byte] for byte in b]
            while len(ids) >= 2:
                stats = {}
                for pair in zip(ids,ids[1:]):
                    stats[pair] = stats.get(pair,0) + 1
                pair = min(stats,key=lambda p : merges.get(p,float("inf")))
                if pair not in merges: break

                new_idx = merges[pair]
                ids = self._merge(ids,pair,new_idx)
            assert len(ids) == 2
            merges[(ids[0],ids[1])] = rank
        return merges

if __name__ == "__main__":
    tokenizer = GPT4Tokenizer()

    # print a few recovered merges to sanity check they look reasonable
    print("\nfirst 5 recovered merge rules:")
    for pair, idx in list(tokenizer.merges.items())[:5]:
        p0_bytes = tokenizer.vocab[pair[0]]
        p1_bytes = tokenizer.vocab[pair[1]]
        combined_bytes = tokenizer.vocab[idx]
        print(f"id {pair[0]} ({p0_bytes}) + id {pair[1]} ({p1_bytes}) -> id {idx} ({combined_bytes})")

