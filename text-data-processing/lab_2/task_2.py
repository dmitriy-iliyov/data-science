from nltk import *
from nltk.corpus import brown


def do():
    # a
    adventure_texts = brown.fileids(categories='adventure')
    tenth_adventure_words_list = brown.words(adventure_texts[9])
    tenth_adventure_text = ' '.join(tenth_adventure_words_list)

    with open('files/10_adventure_text.txt', 'w') as file:
        file.write(tenth_adventure_text)
    tenth_adventure_sentence = sent_tokenize(tenth_adventure_text)
    tenth_adventure_words_tokenize = []
    whitespace_wt = WhitespaceTokenizer()
    for sentence in tenth_adventure_sentence[:-1]:
        tenth_adventure_words_tokenize += whitespace_wt.tokenize(sentence)
    with open('files/new_10_adventure_text.txt', 'w') as file:
        for sentence in tenth_adventure_sentence[:-1]:
            print(sentence)
            file.write(sentence)

    # b
    pos_tagged = pos_tag(tenth_adventure_words_list, tagset="universal")
    for pair in pos_tagged:
        if pair[1] == "ADJ":
            tenth_adventure_words_tokenize.remove(pair[0])
    tenth_adventure_ADJ_deleted_text = ' '.join(tenth_adventure_words_tokenize)
    print("\nречення без прикметників")
    print(tenth_adventure_ADJ_deleted_text)

    with open('files/10_adventure_ADJ_deleted_text.txt', 'w') as file:
        file.write(tenth_adventure_ADJ_deleted_text)


