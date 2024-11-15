import sys

from nltk import *
import spacy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/')


def _prepare_data():
    with open('files/lab6-2.txt') as file:
        lines = [line.strip() for line in file]
    return ''.join(lines)


class Task2:
    def __init__(self):
        self.file_data = _prepare_data()

    def do_task(self):
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(self.file_data)

        adjectives = []
        not_stop_words = []

        for token in doc:
            token.text.split()
            if not token.is_stop and not token.is_punct and not token.like_num:
                not_stop_words.append(token.text)
                if token.pos_ == "ADJ":
                    adjectives.append(token.text)

        numbers_and_persons = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in ["CARDINAL", "DATE", "PERSON"]]
        print("Not stop words:")
        print(not_stop_words)
        print("ADJs:")
        print(adjectives)
        print("Numbers and Persons")
        print(numbers_and_persons)
