import regex as re


class BasicTokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

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

    def decode(self, ids):
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        return text_bytes.decode("utf-8", errors="replace")

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

    def _get_stats(self, ids):
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts


GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class RegexTokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.compiled_pattern = re.compile(GPT4_SPLIT_PATTERN)
        self.special_tokens = {}
        self.inverse_special_tokens = {}

    def register_special_tokens(self, special_tokens):
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

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

    def decode(self, ids):
        part_bytes = []
        for idx in ids:
            if idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                part_bytes.append(self.vocab[idx])
        text_bytes = b"".join(part_bytes)
        return text_bytes.decode("utf-8", errors="replace")


def main():
    with open("taylorswift.txt", "r", encoding="utf-8") as f:
        text = f.read()

    print("--- basic tokenizer ---")
    basic = BasicTokenizer()
    basic.train(text, vocab_size=260, verbose=True)

    print("\n--- regex tokenizer ---")
    regex_tok = RegexTokenizer()
    regex_tok.train(text, vocab_size=260, verbose=True)

    test = "hello world! 123"
    print("\ntest:", test)
    print("basic:", basic.encode(test))
    print("regex:", regex_tok.encode(test))


if __name__ == "__main__":
    main()
