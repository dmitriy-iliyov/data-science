import numpy as np
import pandas as pd


def to_tf_idf(prepared_corpus):
    from sklearn.feature_extraction.text import TfidfVectorizer
    tv = TfidfVectorizer(min_df=0., max_df=1., norm='l2', use_idf=True, smooth_idf=True)
    tv_matrix = tv.fit_transform(prepared_corpus)
    return pd.DataFrame(np.round(tv_matrix.toarray(), 2), columns=tv.get_feature_names_out())


def clusterizate(prepared_corpus):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import AgglomerativeClustering
    tv = TfidfVectorizer(min_df=0., max_df=1., norm='l2', use_idf=True, smooth_idf=True)
    tfidf_matrix = tv.fit_transform(prepared_corpus)
    clustering = AgglomerativeClustering(n_clusters=3)
    clusters = clustering.fit_predict(tfidf_matrix.toarray())
    return clusters
