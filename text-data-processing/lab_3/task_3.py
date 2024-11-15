from nltk import WordPunctTokenizer
from gensim.models import word2vec


def to_Word2Vec(prepared_corpus):
    wpt = WordPunctTokenizer()
    prepared_corpus_words = [wpt.tokenize(sentence) for sentence in prepared_corpus]
    w2v_model = word2vec.Word2Vec(prepared_corpus_words, vector_size=10, window=10, min_count=1, sample=1e-3)
    return w2v_model


def task_3(w2v_model):
    print("most similar words to architecture:", w2v_model.wv.most_similar('architecture'))
    print("most similar words to biscotti:", w2v_model.wv.most_similar('biscotti'))
