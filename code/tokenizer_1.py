import re
from collections import defaultdict, Counter
from heapq import heapify, heappop, heappush
from typing import List, Tuple
from base_tokenizer import BaseTokenizer



class BPETokenizer(BaseTokenizer):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.merges = []
        self.merge_ranks = {}

    def train(self, texts: List[str]) -> None:

        bigrams = self.find_bigrams(texts)
        max_bigram_length = max(len((pair[0] + pair[1]).encode("utf-8")) for pair in bigrams)

        texts = preprocess_texts(texts)
        byte_texts = [t.encode('utf-8') for t in texts]

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


        # STEP 2: initialize token vocab
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
            if not pair_heap:
                break

            freq, pair = heappop(pair_heap)
            freq = -freq

            if pair_freqs[pair] != freq:
                continue

            new_token = pair[0] + pair[1]
            if len(new_token) > max_bigram_length + 5:
                continue

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


    def find_bigrams(self, text):
        """
        Given a list of sentence strings, finds the most and least common word bigrams.
        Prints top 5 and bottom 5, and returns all bigrams.
        """
        bigrams = []

        for sentence in text:
            words = sentence.strip().split()
            for i in range(len(words) - 1):
                bigram = (words[i], words[i + 1])
                bigrams.append(bigram)

        return bigrams

    def is_bigram_merged(self, merges, byte_bigrams):
        """
        Check whether at least one of the input byte bigrams appears in the merges.
        """
        merge_set = set(merges)  # Merges are tuples of (bytes, bytes)

        for w1, w2 in byte_bigrams:
            if (w1, w2) in merge_set:
                return True
            if (w1 + b' ', w2) in merge_set:
                return True
            if (w1, b' ' + w2) in merge_set:
                return True
            if (w1 + b' ', w2 + b' ') in merge_set:
                return True

        return False


# Cleanup 1: Remove URLs starting with http and ending with space
def remove_urls(text):
    return re.sub(r'http\S+\s', '', text)


# Cleanup 2: Remove HTML entities like &amp;
def remove_html_entities(text):
    return re.sub(r'&[^;\s]+;', '', text)


# Cleanup 3: Replace repeated symbols with a single instance
def collapse_symbols(text):
    return re.sub(r'([^\w\s])\1+', r'\1', text)


# Cleanup 4: Replace repeated letters (more than twice) with two instances
def limit_repeated_letters(text):
    return re.sub(r'([a-zA-Z])\1{2,}', r'\1\1', text)


# Cleanup 5: Remove @mention if it is at the beginning of the sentence
def remove_leading_mentions(text):
    return re.sub(r'^@\S+\s+', '', text)


def preprocess_texts(texts, cleanups_to_apply=None):
    all_cleanups = [
        remove_urls,
        remove_html_entities,
        collapse_symbols,
        limit_repeated_letters,
        remove_leading_mentions
    ]

    # If no specific cleanups provided, apply all
    if cleanups_to_apply is None:
        cleanups_to_apply = all_cleanups

    processed_texts = []
    for text in texts:
        for cleanup in cleanups_to_apply:
            text = cleanup(text)
        processed_texts.append(text)

    return processed_texts
