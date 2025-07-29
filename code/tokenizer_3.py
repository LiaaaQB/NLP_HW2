import re
from collections import defaultdict, Counter
from heapq import heapify, heappop, heappush
from typing import List, Tuple
from base_tokenizer import BaseTokenizer
import time
from text_cleaning import preprocess_texts



# TODO: change to UTF-8 encoding
# TODO: apply heap to utf-encoding method
# DONE get stats on most common bigrams
# DONE make sure at least one bigram makes it to the vocab.



class BPETokenizer(BaseTokenizer):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.merges = []

    def train(self, texts: List[str]) -> None:

        # STEP 2: preprocess as strings, then convert to bytes
        texts = preprocess_texts(texts, mode="general")
        byte_texts = [t.encode('utf-8') for t in texts]

        vocab = []
        sentences = []
        pair_freqs = Counter()
        pair_to_words = defaultdict(set)

        for idx, byte_line in enumerate(byte_texts):
            sen = [bytes([b]) for b in byte_line.strip()]
            sentences.append([sen, 1])

            for i in range(len(sen) - 1):
                pair = (sen[i], sen[i + 1])
                pair_freqs[pair] += 1
                pair_to_words[pair].add(len(sentences) - 1)

        # STEP 2b: initialize token vocab
        next_token_id = len(self.token_to_id)
        for word, _ in sentences:
            for token in word:
                if token not in self.token_to_id:
                    self.token_to_id[token] = next_token_id
                    self.id_to_token[next_token_id] = token
                    next_token_id += 1

        # STEP 3: build heap
        pair_heap = [(-freq, pair) for pair, freq in pair_freqs.items()]
        heapify(pair_heap)

        # STEP 4: BPE loop
        iteration = 1
        while len(self.token_to_id) < self.vocab_size:
            start_time = time.time()
            if not pair_heap:
                break

            freq, pair = heappop(pair_heap)
            freq = -freq

            if pair_freqs[pair] != freq:
                continue

            new_token = pair[0] + pair[1]

            self.merges.append(pair)
            self.token_to_id[new_token] = next_token_id
            self.id_to_token[next_token_id] = new_token
            next_token_id += 1

            affected_word_ids = pair_to_words[pair]
            pair_freqs[pair] = 0
            pair_to_words[pair] = set()
            pending_pairs = dict()

            for idx in affected_word_ids:
                word, count = sentences[idx]
                i = 0
                while i < len(word) - 1:
                    if (word[i], word[i + 1]) == pair:
                        word[i:i + 2] = [new_token]

                        if i > 0:
                            prev_pair = (word[i - 1], new_token)
                            pair_freqs[prev_pair] += count
                            pair_to_words[prev_pair].add(idx)
                            pending_pairs[prev_pair] = pair_freqs[prev_pair]

                        if i < len(word) - 1:
                            next_pair = (new_token, word[i + 1])
                            pair_freqs[next_pair] += count
                            pair_to_words[next_pair].add(idx)
                            pending_pairs[next_pair] = pair_freqs[next_pair]
                    else:
                        i += 1

            for updated_pair, updated_freq in pending_pairs.items():
                heappush(pair_heap, (-updated_freq, updated_pair))

            end_time = time.time()
            if iteration % 50 == 0:
                print(f"completed iteration: {iteration} in {end_time - start_time:.4f} seconds, added {new_token}")
            iteration += 1




    def encode(self, text: str) -> List[int]:
        tokens = [bytes([b]) for b in text.encode('utf-8')]
        merges_set = set(self.merges)

        while True:
            pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
            merge_found = False
            for i, pair in enumerate(pairs):
                if pair in merges_set:
                    merged = pair[0] + pair[1]
                    tokens = tokens[:i] + [merged] + tokens[i + 2:]
                    merge_found = True
                    break
            if not merge_found:
                break

        return [self.token_to_id.get(tok, self.token_to_id.get(b"[UNK]", 1)) for tok in tokens]

    def decode(self, token_ids: List[int]) -> str:
        tokens = [
            b"[UNK]" if i == 1 else self.id_to_token.get(i, b"[UNK]")
            for i in token_ids
        ]
        return b''.join(tokens).decode("utf-8", errors="replace")





