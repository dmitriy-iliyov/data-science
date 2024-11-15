import json
import pandas as pd
import spacy
from spacy.training.example import Example
from spacy.util import minibatch, compounding


class Task2:
    def __init__(self):
        self.df = self._prepare_data()
        self.train_data = [(row['utterance'], {'cats': {row['intent']: 1.0}}) for _, row in self.df.iterrows()]
        self._learn_model()

    def _prepare_data(self):
        with open('files/media.json', 'r') as file:
            data = json.load(file)

        utterances = []
        intents = []
        for dialogue in data:
            for turn in dialogue['turns']:
                if turn['speaker'] == 'USER':
                    utterance = turn['utterance']
                    intent = turn['frames'][0]['state']['active_intent']
                    utterances.append(utterance)
                    intents.append(intent)

        return pd.DataFrame({'utterance': utterances, 'intent': intents})

    def _learn_model(self):
        self.nlp = spacy.blank('en')
        textcat = self.nlp.add_pipe('textcat', last=True)
        for intent in self.df['intent'].unique():
            textcat.add_label(intent)
        self.nlp.begin_training()
        n_iter = 10
        batch_sizes = compounding(4.0, 32.0, 1.001)
        for i in range(n_iter):
            batches = minibatch(self.train_data, size=batch_sizes)
            for batch in batches:
                texts, annotations = zip(*batch)
                examples = [Example.from_dict(self.nlp.make_doc(text), annotation) for text, annotation in
                            zip(texts, annotations)]
                self.nlp.update(examples, drop=0.5)

        self.nlp.to_disk('intent_model')
        self.nlp = spacy.load('intent_model')

    def do(self, text):
        return [(utterance, self.nlp(utterance).cats) for utterance in text]
