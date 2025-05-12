from preprocessing import extract_sentences, get_token_to_index, tokenize_sents, extract_sentences_tsv
import glob
from metrics import compute_aer
from models import WordAligner, WordPositionAligner

import numpy as np

all_sentences = []
all_targets = []
for f in glob.iglob('./CzEnAli_1.0/merged_data/*/*.wa'):
    a, b = extract_sentences(f, normalize=True)
    all_sentences.extend(a)
    all_targets.extend(b)

test_sentences = all_sentences.copy()

a = extract_sentences_tsv("./europarl-v10.cs-en.pair.tsv", normalize=True)[:5000]
all_sentences.extend(a)
t_idx_src, t_idx_tgt = get_token_to_index(all_sentences)
tokenized_sentences = tokenize_sents(all_sentences, t_idx_src, t_idx_tgt)

test_idx_src, test_idx_tgt = get_token_to_index(test_sentences)
test_tokenized_sentences = tokenize_sents(test_sentences, test_idx_src, test_idx_tgt)

word_aligner = WordAligner(len(t_idx_src), len(t_idx_tgt), 20)
history = word_aligner.fit(tokenized_sentences)
print(compute_aer(all_targets, word_aligner.align(tokenized_sentences)))
