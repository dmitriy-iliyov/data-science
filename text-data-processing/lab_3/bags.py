import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(prepared_corpus):
    cv = CountVectorizer(min_df=0., max_df=1.)
    matrix = cv.fit_transform(prepared_corpus)
    return pd.DataFrame(matrix.toarray(), columns=cv.get_feature_names_out())


def bag_of_bigrams(prepared_corpus):
    cv = CountVectorizer(ngram_range=(2, 2))
    matrix = cv.fit_transform(prepared_corpus)
    return pd.DataFrame(matrix.toarray(), columns=cv.get_feature_names_out())


def bigram_phrase_vector(prepared_corpus, phrase):
    cv = CountVectorizer(ngram_range=(2, 2))
    df = pd.DataFrame(cv.fit_transform(prepared_corpus).toarray(), columns=cv.get_feature_names_out())
    vector = df[phrase.lower()].tolist()
    return vector
