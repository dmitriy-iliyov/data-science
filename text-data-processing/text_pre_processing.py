import re
import numpy as np
import pandas as pd
from nltk import *
from nltk.corpus import stopwords


def _start_pre_processing(doc):
    doc = re.sub(r'http[s]?://\S+|www\.\S+', '', doc)
    doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I | re.A)
    doc = doc.lower()
    doc = doc.strip()
    wpt = WordPunctTokenizer()
    tokens = wpt.tokenize(doc)
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]
    doc = ' '.join(filtered_tokens)
    return doc


def _pdSeries_pre_processing(series):
    prepared_corpus = series.apply(lambda x: _start_pre_processing(x))
    return prepared_corpus


def _str_pre_processing(_str):
    sentences = _str.split('.')
    prepared_corpus = [_start_pre_processing(sentence) for sentence in sentences]
    prepared_corpus = ' '.join(list(filter(None, prepared_corpus)))
    return prepared_corpus


def _list_pre_processing(_list):
    return [_start_pre_processing(element) for element in doc]


def do_pre_processing(doc):
    if isinstance(doc, pd.Series):
        return _pdSeries_pre_processing(doc)
    elif isinstance(doc, str):
        return _str_pre_processing(doc)
    elif isinstance(doc, list):
        return _list_pre_processing(doc)
    else:
        print("ERROR:   TextPreProcessor can't prepare this type of data.")
        return None

