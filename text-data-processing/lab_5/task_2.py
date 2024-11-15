import os
import sys
from operator import itemgetter

import nltk

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/')

import text_pre_processing as tpp


def _pre_processing_data_for_task_2(doc):
    return tpp.do_pre_processing(doc[:10000])


def _compute_ngrams(sequence, n):
    return list(zip(*(sequence[index:] for index in range(n))))


def _get_top_ngrams(corpus, ngram_val=1, limit=5):
    tokens = nltk.word_tokenize(corpus)
    ngrams = _compute_ngrams(tokens, ngram_val)
    ngrams_freq_dist = nltk.FreqDist(ngrams)
    sorted_ngrams = sorted(ngrams_freq_dist.items(), key=itemgetter(1), reverse=True)[:limit]
    sorted_ngrams = [(' '.join(text), freq) for text, freq in sorted_ngrams]
    return sorted_ngrams


def do_task_2(text):
    prepared_corpus_1 = _pre_processing_data_for_task_2(text)
    return _get_top_ngrams(corpus=prepared_corpus_1, ngram_val=3, limit=10)
