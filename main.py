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


if __name__ == "__main__":
    with open("taylorswift.txt", "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = BasicTokenizer()
    tokenizer.train(text, vocab_size=276, verbose=True)

    print(tokenizer.decode(tokenizer.encode("hello world!")))
