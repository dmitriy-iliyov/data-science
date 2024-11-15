import json

import spacy
from spacy.matcher import Matcher


class TaskSolver:
    def __init__(self):
        self._nlp = spacy.load("en_core_web_md")
        self._read_data()
        self._prepare_data()

    def _read_data(self):
        with open('files/media.json', 'r') as file:
            self._json_data = json.load(file)

    def _prepare_data(self):
        _service_results = []
        _user_utterances = []
        _speakers_utterances = []

        for dialogue in self._json_data:
            for turn in dialogue["turns"]:
                if turn['speaker'] == "USER" or turn['speaker'] == "SYSTEM":
                    _speakers_utterances.append(f"{turn['utterance']}")
                    if turn['speaker'] == "USER":
                        _user_utterances.append(f"{turn['utterance']}")

                for frame in turn["frames"]:
                    if "service_results" in frame:
                        for result in frame["service_results"]:
                            result_text = ' '.join(f'{k}: {v};' for k, v in result.items())
                            _service_results.append(result_text)

        self._task_1_doc = self._nlp(' '.join(_service_results))
        self._task_2_doc = self._nlp(' '.join(_user_utterances))
        self._task_3_doc = self._nlp(' '.join(_speakers_utterances))

    def do_task_1(self):

        # task 1 Виділити жанри фільмів за допомогою класу Matcher.

        _all_genres = []
        matcher = Matcher(self._nlp.vocab)

        pattern_1 = [{"LOWER": "genre"},
                     {"IS_PUNCT": True},
                     {}]

        matcher.add("film_genres", [pattern_1])
        matches = matcher(self._task_1_doc)
        for match_id, start, end in matches:
            m_span = self._task_1_doc[start:end]
            print(start, end, m_span.text)
            if m_span[2].text not in _all_genres:
                _all_genres.append(m_span[2].text)

        print("all film genres mentioned in dialogues:")
        print(_all_genres)

    def do_task_2(self):

        # task 2 Виділити висловлювання користувача, що є підтвердженнями (наприклад, Yes I do), за допомогою шаблонів.

        print(self._task_2_doc)

        matcher = Matcher(self._nlp.vocab)

        patterns = [
            [{"LOWER": {"IN": ["yes", "yeah", "yep", "sure", "absolutely"]}}],
            [{"LOWER": "yes"}, {"LOWER": "i"}, {"LOWER": {"IN": ["like", "do", "love"]}}],
            [{"LOWER": "i"}, {"LOWER": {"IN": ["like", "love"]}}],
            [{"LOWER": "yeah"}, {"LOWER": "that"}, {"LOWER": {"IN": ["'s", "is"]}}, {"LOWER": "correct"}]
        ]

        for pattern in patterns:
            matcher.add("confirmation", [pattern])

        matches = matcher(self._task_2_doc)

        for match_id, start, end in matches:
            m_span = self._task_2_doc[start:end]
            print(start, end, m_span.text)

    def do_task_3(self):

        # task 3 виділення намірів за допомогою синтаксичних залежностей

        intense = []
        for sentence in self._task_3_doc.sents:
            for token in sentence:
                if token.dep_ == "dobj":
                    tmp = {"verb": token.head.text,
                           "direct object": token.text,
                           "conjunctions": [t.text for t in token.conjuncts]}
                    intense.append(tmp)
                    print(tmp)
