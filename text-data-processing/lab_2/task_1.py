from nltk import *
from nltk.stem.snowball import EnglishStemmer


def do(content):
    # a
    divided_to_sent = sent_tokenize(content)
    divided_to_words = []
    whitespace_wt = WhitespaceTokenizer()
    for sent in divided_to_sent:
        divided_to_words += whitespace_wt.tokenize(sent)
    word_count = len(divided_to_words)
    print("word count = " + str(word_count))

    # b
    print("частини мови")
    pos_tagged = pos_tag(divided_to_words)
    print(pos_tagged[1:])

    # c
    print("корені слів")
    second_sentence_to_words = regexp_tokenize(divided_to_sent[1], pattern='\w+')
    stemmed_list = []
    sst = EnglishStemmer()
    for word in second_sentence_to_words:
        stemmed_list.append(sst.stem(word))
    result = " ".join(stemmed_list)
    print(result)
