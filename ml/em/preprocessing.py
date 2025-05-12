from dataclasses import dataclass
from typing import Dict, List, Tuple
import xml.etree.ElementTree as ET
import re
from collections import Counter
import unicodedata
import csv
import pandas as pd

import numpy as np


@dataclass(frozen=True)
class SentencePair:
    """
    Contains lists of tokens (strings) for source and target sentence
    """
    source: List[str]
    target: List[str]


@dataclass(frozen=True)
class TokenizedSentencePair:
    """
    Contains arrays of token vocabulary indices (preferably np.int32) for source and target sentence
    """
    source_tokens: np.ndarray
    target_tokens: np.ndarray


@dataclass(frozen=True)
class LabeledAlignment:
    """
    Contains arrays of alignments (lists of tuples (source_pos, target_pos)) for a given sentence.
    Positions are numbered from 1.
    """
    sure: List[Tuple[int, int]]
    possible: List[Tuple[int, int]]


def extract_sentences(filename: str, normalize=False) -> Tuple[List[SentencePair], List[LabeledAlignment]]:
    """
    Given a file with tokenized parallel sentences and alignments in XML format, return a list of sentence pairs
    and alignments for each sentence.

    Args:
        filename: Name of the file containing XML markup for labeled alignments
        normalize: Apply unicode-normalization and low case

    Returns:
        sentence_pairs: list of `SentencePair`s for each sentence in the file
        alignments: list of `LabeledAlignment`s corresponding to these sentences
    """

    with open(filename, 'r',
              encoding='utf-8') as f:  # Here I asked deepseek "How can i fix ampersant error without editing original file"
        xml_content = f.read()

    sanitized = re.sub(r'&(?!(amp|lt|gt|apos|quot|#\d+);)', '&amp;', xml_content)

    sentence_pairs = []
    aligments = []

    root = ET.fromstring(sanitized)

    for sentence in root.findall('s'):
        eng = sentence.find('english').text.split()
        che = sentence.find('czech').text.split()
        if normalize:
            eng = [unicodedata.normalize("NFC", s).lower() for s in eng]
            che = [unicodedata.normalize("NFC", s).lower() for s in che]
        sentence_pairs.append(SentencePair(eng, che))

        s = sentence.find('sure').text
        if s is not None:
            sure = [tuple(map(int, i.split('-'))) for i in s.split()]
        else:
            sure = []

        p = sentence.find('possible').text
        if p is not None:
            possible = [tuple(map(int, i.split('-'))) for i in p.split()]
        else:
            possible = []

        aligments.append(LabeledAlignment(sure, possible))

    return sentence_pairs, aligments


def extract_sentences_tsv(filename: str, normalize=False, colab=True) -> Tuple[List[SentencePair]]:
    """
    Given a file with tokenized parallel sentences and alignments in TSV format, return a list of sentence pairs
    and alignments for each sentence.

    Args:
        filename: Name of the file containing TSV markup for labeled alignments
        normalize: Apply unicode-normalization and low case
        colab: You are working in piece of dog sh*t called Google colab
    Returns:
        sentence_pairs: list of `SentencePair`s for each sentence in the file
    """
    if colab:
        data = pd.read_csv(filename, sep='\t', on_bad_lines='skip', header=None, dtype='str')[[0, 1]].dropna()

        sentence_pairs = []
        for row in range(data.shape[0]):
            eng = data.iloc[row, 1].split()
            che = data.iloc[row, 0].split()
            if normalize:
                eng = [unicodedata.normalize("NFC", s).lower() for s in eng]
                che = [unicodedata.normalize("NFC", s).lower() for s in che]
            if len(eng) == 0 or len(che) == 0:
                continue
            sentence_pairs.append(SentencePair(eng, che))

        return sentence_pairs

    with open(filename, 'r',
              encoding='utf-8') as fd:
        rd = csv.reader(fd, delimiter="\t", quotechar='"')
        sentence_pairs = []
        for row in rd:
            eng = row[1].split()
            che = row[0].split()
            if normalize:
                eng = [unicodedata.normalize("NFC", s).lower() for s in eng]
                che = [unicodedata.normalize("NFC", s).lower() for s in che]
            if len(eng) == 0 or len(che) == 0:
                continue
            sentence_pairs.append(SentencePair(eng, che))

        return sentence_pairs


def get_token_to_index(sentence_pairs: List[SentencePair], freq_cutoff=None) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Given a parallel corpus, create two dictionaries token->index for source and target language.

    Args:
        sentence_pairs: list of `SentencePair`s for token frequency estimation
        freq_cutoff: if not None, keep only freq_cutoff most frequent tokens in each language

    Returns:
        source_dict: mapping of token to a unique number (from 0 to vocabulary size) for source language
        target_dict: mapping of token to a unique number (from 0 to vocabulary size) target language

    """

    source_counter = Counter()
    target_counter = Counter()

    for pair in sentence_pairs:
        source_counter.update(pair.source)
        target_counter.update(pair.target)

    if freq_cutoff is not None:
        source_words = [i[0] for i in source_counter.most_common(freq_cutoff)]
        target_words = [i[0] for i in target_counter.most_common(freq_cutoff)]
    else:
        source_words = list(source_counter)
        target_words = list(target_counter)

    source_dict = {j: i for i, j in enumerate(source_words)}
    target_dict = {j: i for i, j in enumerate(target_words)}

    return source_dict, target_dict


def tokenize_sents(sentence_pairs: List[SentencePair], source_dict, target_dict) -> List[TokenizedSentencePair]:
    """
    Given a parallel corpus and token_to_index for each language, transform each pair of sentences from lists
    of strings to arrays of integers. If either source or target sentence has no tokens that occur in corresponding
    token_to_index, do not include this pair in the result.
    
    Args:
        sentence_pairs: list of `SentencePair`s for transformation
        source_dict: mapping of token to a unique number for source language
        target_dict: mapping of token to a unique number for target language

    Returns:
        tokenized_sentence_pairs: sentences from sentence_pairs, tokenized using source_dict and target_dict
    """
    tokenized_sentence_pairs = []

    def dict_process(s, d):
        if s in d:
            return d[s]
        else:
            return np.nan

    for pair in sentence_pairs:
        source_tokens = np.array([dict_process(i, source_dict) for i in pair.source])
        target_tokens = np.array([dict_process(i, target_dict) for i in pair.target])
        if np.any(np.isnan(source_tokens)) or np.any(np.isnan(target_tokens)):
            continue
        tokenized_sentence_pairs.append(TokenizedSentencePair(source_tokens, target_tokens))

    return tokenized_sentence_pairs
