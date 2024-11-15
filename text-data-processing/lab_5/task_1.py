import os
import sys

import gensim
import nltk
import pandas as pd
from nltk.corpus import twitter_samples

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/')

import text_pre_processing as tpp


def _prepare_data_frame_for_task_1():
    with open('files/bbc-news-data.csv') as file:
        file_data = [line.strip() for line in file]
    df_column_names = file_data[0].split('\t')
    df_lines = [line.split('\t') for line in file_data[1:]]
    return pd.DataFrame(df_lines, columns=df_column_names)


def _pre_processing_df_for_task_1():
    data_df = _prepare_data_frame_for_task_1()
    data_df['clear content'] = tpp.do_pre_processing(data_df['content'])
    return data_df


def _download_twitter():
    nltk.download('twitter_samples')


def do_task_1():
    df = _pre_processing_df_for_task_1()
    words_from_clear_content = [nltk.word_tokenize(line) for line in df['clear content'].values]
    bigram = gensim.models.Phrases(words_from_clear_content, min_count=5, threshold=5, delimiter='-')
    bigram_model = gensim.models.phrases.Phraser(bigram)
    norm_corpus_bigrams = [bigram_model[doc] for doc in df['clear content']]
    dictionary = gensim.corpora.Dictionary(norm_corpus_bigrams)
    dictionary.filter_extremes(no_below=10, no_above=0.6)
    bow_corpus = [dictionary.doc2bow(text) for text in norm_corpus_bigrams]
    Total_topics = 10
    lsi_bow = gensim.models.LsiModel(bow_corpus, id2word=dictionary, num_topics=Total_topics, onepass=True,
                                     chunksize=1740, power_iters=1000)

    for topic_id, topic in lsi_bow.show_topics():
        print(f"Topic {topic_id + 1}: {topic}")
    print("\n")

    positive_tweets = ''.join(twitter_samples.strings('positive_tweets.json')[:500])
    negative_tweets = ''.join(twitter_samples.strings('negative_tweets.json')[:500])
    p_and_n_tweets = ''.join(twitter_samples.strings('positive_tweets.json')[501:751] +
                             twitter_samples.strings('negative_tweets.json')[501:751])
    docs = [positive_tweets, negative_tweets, p_and_n_tweets]
    prepared_docs = [tpp.do_pre_processing(doc) for doc in docs]

    new_corpus_bigrams = [bigram_model[doc] for doc in prepared_docs]
    new_corpus = [dictionary.doc2bow(text) for text in new_corpus_bigrams]

    new_topics = lsi_bow[new_corpus]

    for i, doc_topics in enumerate(new_topics):
        print(f"Document {i + 1}:")
        for topic_id, weight in doc_topics:
            print(f"Topic {topic_id +1}: weight = {weight}")
