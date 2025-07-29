import re
from typing import List

# Define cleanup steps
def safe_decode(text: str) -> str:
    if isinstance(text, bytes):
        return text.decode('utf-8', errors='replace')
    return str(text)

def normalize_case_and_strip(text: str) -> str:
    return text.lower().strip()

def remove_control_chars(text: str) -> str:
    return re.sub(r'[\x00-\x1f\x7f]', '', text)

def remove_emojis(text: str) -> str:
    return re.sub(r'[\U00010000-\U0010ffff]', '', text)

def remove_urls(text: str) -> str:
    return re.sub(r'https?://\S+|www\.\S+', '', text)

def remove_mentions(text: str) -> str:
    return re.sub(r'@\w+', '', text)

def collapse_repeated_punct(text: str) -> str:
    return re.sub(r'([^\w\s])\1{1,}', r'\1', text)

def limit_repeated_letters(text: str) -> str:
    return re.sub(r'([a-zA-Z])\1{2,}', r'\1\1', text)

def replace_emojis(text: str) -> str:
    return re.sub(r'[\U00010000-\U0010ffff]', '', text)

def replace_urls(text: str) -> str:
    return re.sub(r'https?://\S+|www\.\S+', '', text)

def replace_mentions(text: str) -> str:
    return re.sub(r'@\w+', '', text)

def cap_repeated_punct(text: str, max_rep: int = 3) -> str:
    return re.sub(r'([^\w\s])\1{%d,}' % (max_rep - 1), r'\1' * max_rep, text)

def cap_repeated_letters(text: str, max_rep: int = 3) -> str:
    return re.sub(r'([a-zA-Z])\1{%d,}' % (max_rep - 1), r'\1' * max_rep, text)


# Define modes
CLEANUP_MODES = {
    "general": [
        safe_decode,
        remove_control_chars,
        replace_urls,
        replace_mentions,
        replace_emojis,
        cap_repeated_punct,
        cap_repeated_letters,
        str.strip
    ],
    "social": [
        safe_decode,
        remove_urls,
        remove_mentions,
        collapse_repeated_punct,
        limit_repeated_letters
    ],

}

# Master function
def preprocess_texts(texts: List[str], mode: str = "general") -> List[str]:
    if mode not in CLEANUP_MODES:
        raise ValueError(f"Unknown mode: {mode}. Available: {list(CLEANUP_MODES.keys())}")

    steps = CLEANUP_MODES[mode]
    cleaned = []
    for text in texts:
        for step in steps:
            text = step(text)
        cleaned.append(text)
    return cleaned
