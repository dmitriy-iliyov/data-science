import random

import spacy
from spacy.training import Example


class Task1:
    def __init__(self):
        self._nlp = spacy.load("en_core_web_md")
        self._trainset = self._set_train_data()
        self._add_new_ner()
        self._learn_model()

    def _set_train_data(self):
        return [
            ("He won the Nobel Prize in Physics", {"entities": [(11, 22, "AWARD")]}),
            ("She received the Pulitzer Prize for Fiction", {"entities": [(16, 30, "AWARD")]}),
            ("They awarded him the Academy Award", {"entities": [(21, 34, "AWARD")]}),
            ("The Grammy Award goes to her", {"entities": [(4, 16, "AWARD")]}),
            ("He was nominated for an Emmy Award", {"entities": [(22, 33, "AWARD")]}),
            ("She won the Golden Globe for Best Actress", {"entities": [(8, 20, "AWARD")]}),
            ("He received a Tony Award for his performance", {"entities": [(14, 24, "AWARD")]}),
            ("The Booker Prize was given to the author", {"entities": [(4, 16, "AWARD")]}),
            ("She was honored with the BAFTA Award", {"entities": [(23, 34, "AWARD")]}),
            ("He won the Fields Medal in Mathematics", {"entities": [(8, 20, "AWARD")]}),
            ("The Edgar Award was presented to the best mystery novel", {"entities": [(4, 15, "AWARD")]}),
            ("She clinched the MTV Music Award for Best New Artist", {"entities": [(17, 33, "AWARD")]}),
            ("He was recognized with the Peabody Award for his outstanding journalism",
             {"entities": [(24, 37, "AWARD")]}),
            ("The Hugo Award for Best Novel went to a science fiction writer", {"entities": [(4, 15, "AWARD")]}),
            ("She received the Critics' Choice Movie Award for Best Actress", {"entities": [(16, 41, "AWARD")]}),
            ("He was honored with the MacArthur Fellowship", {"entities": [(23, 41, "AWARD")]}),
            ("The Man Booker International Prize was awarded to a celebrated author", {"entities": [(4, 33, "AWARD")]}),
            ("The National Book Award for Fiction was presented last night", {"entities": [(4, 30, "AWARD")]}),
            ("She accepted the Goya Award for Best Director", {"entities": [(17, 27, "AWARD")]}),
            ("He was the recipient of the Nebula Award for Best Short Story", {"entities": [(23, 35, "AWARD")]}),
            ("The Cannes Film Festival awarded the Palme d'Or to a groundbreaking movie",
             {"entities": [(29, 38, "AWARD")]}),
            ("She won the Critics' Circle Theatre Award for Best Actress", {"entities": [(8, 38, "AWARD")]}),
            ("He took home the ESPY Award for Best Male Athlete", {"entities": [(14, 24, "AWARD")]}),
            ("The Man Asian Literary Prize was given to an emerging author", {"entities": [(4, 29, "AWARD")]}),
            ("She was awarded the Order of Merit for her contributions", {"entities": [(17, 30, "AWARD")]}),
            ("He received the Copley Medal for his scientific achievements", {"entities": [(14, 26, "AWARD")]}),
            ("The James Tait Black Memorial Prize was awarded to a novelist", {"entities": [(4, 35, "AWARD")]}),
            ("She was the laureate of the John Bates Clark Medal", {"entities": [(23, 47, "AWARD")]}),
            ("He earned the Avery Fisher Prize for his excellence in music", {"entities": [(11, 27, "AWARD")]}),
            ("The Heisman Trophy was awarded to the best college football player", {"entities": [(4, 18, "AWARD")]}),
        ]

    def _add_new_ner(self):
        if "ner" not in self._nlp.pipe_names:
            self._ner = self._nlp.add_pipe("ner", last=True)
        else:
            self._ner = self._nlp.get_pipe("ner")
        self._ner.add_label("AWARD")

    def _learn_model(self):
        other_pipes = [pipe for pipe in self._nlp.pipe_names if pipe != "ner"]
        with self._nlp.disable_pipes(*other_pipes):
            optimizer = self._nlp.create_optimizer()
            for i in range(20):
                random.shuffle(self._trainset)
                for text, annotations in self._trainset:
                    doc = self._nlp.make_doc(text)
                    example = Example.from_dict(doc, annotations)
                    self._nlp.update([example], sgd=optimizer)

        self._ner.to_disk("new_ner")

    def do(self, text):
        doc = self._nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]
