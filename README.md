# BPE Tokenizer for Domain-Specific NER

This project implements a Byte Pair Encoding (BPE) tokenizer tailored for Named Entity Recognition (NER) tasks across different domains. The tokenizer is trained and evaluated on three separate datasets with varying characteristics.

## Project Structure

├── data/  
│ ├── ner_data/  
│ │ ├── dev.tagged  
│ │ ├── dev_2_binary.tagged  
│ │ ├── dev_3_binary.tagged  
│ │ ├── train.tagged  
│ │ ├── train_2_binary.tagged  
│ │ ├── train_3_binary.tagged  
│ └── domain_test.txt  
├── code/  
│ ├── base_tokenizer.py  
│ ├── test_tokenizer.py  
│ ├── text_cleaning.py  
│ ├── tokenizer_1.py  
│ ├── tokenizer_2.py  
│ ├── tokenizer_3.py  
│ ├── train_ner_model.py  
│ ├── train_tokenizer.py  
│ ├── train_tokenizer_2.py  
│ └── train_tokenizer_3.py  
├── trained_tokenizer/  
│ ├── tokenizer_1.pkl  
│ ├── tokenizer_2.pkl  
│ └── tokenizer_3.pkl  
├── report.pfd  
└── README.md  

##  Overview

The tokenizer was implemented from scratch with a focus on optimizing runtime and adaptability to different types of domain-specific data. The training process includes cleaning, frequency analysis, and greedy pair merging.

Key features:
- Efficient pair selection using a heap-based frequency structure.
- Custom token frequency updates after each merge.
- Optimized for both tokenization and detokenization.
- Configurable vocabulary size.

##  Domains and Training

### Domain 1: Raw, noisy data
- Preprocessing: Light cleaning (e.g., removal of HTML entities and duplicated letters).
- Tokenizer trained with a vocabulary size of **600**.
- Emphasis on runtime optimization and memory-efficient updates.

### Domain 2: Clean and smaller dataset
- Preprocessing: Minimal (mostly clean input).
- Tokenizer reused from Domain 1.
- Tested different vocabulary sizes but settled on **1000** for optimal F1-score/token ratio.

### Domain 3: Unseen web data
- Preprocessing: Aggressive cleaning (removal of email tags, HTML, repeated characters).
- Combined datasets from Domain 1 and 2 for richer representation.
- Trained tokenizer with a vocabulary size of **5000**.
- Focused on generalization to unfamiliar domains.

##  Evaluation

NER performance was evaluated using F1-score:

| Domain      | Best F1 Score |
|-------------|---------------|
| Domain 1    | 0.41 |
| Domain 2    | 0.95 |
| Domain 3    | ~0.75 (target) |

In Domain 3, despite lack of prior knowledge, the tokenizer showed strong generalization.
