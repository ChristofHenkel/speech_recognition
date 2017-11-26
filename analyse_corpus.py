import pickle
import logging

class CorpusGen:

    def __init__(self, soundcorpus_fp):
        self.fp = soundcorpus_fp

    def gen(self):
        x = []
        y = []
        with open(self.fp, 'rb') as file:
            unpickler = pickle.Unpickler(file)
            while True:
                try:
                    item = unpickler.load()
                    yield item
                except:
                    break


class Analyzer:

    def __init__(self, corpus_gen):
        self.gen = corpus_gen()
        self.data = None

    def get_all_data(self):
        content = []
        for item in self.gen:
            content.append(item)
        self.data = content

    def get_label_distribution(self, data):
        labels = [d['target'] for d in data]
        all_labels = list(set(labels))
        count_dict = {}
        for item in all_labels:
            count_dict[item] = labels.count(item)
        return count_dict


corpus_gen = CorpusGen('assets/corpora/corpus7/train.pm.soundcorpus.p')
analyzer = Analyzer(corpus_gen)
analyzer.get_label_distribution()