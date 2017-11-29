import pickle
import logging
import numpy as np
import sounddevice as sd


class Analyzer:

    def __init__(self, soundcorpus_fp):
        self.fp = soundcorpus_fp
        self.data = None
        self.fs = 16000
        self.gen = self._gen()

    def _gen(self):
        with open(self.fp, 'rb') as file:
            unpickler = pickle.Unpickler(file)
            while True:
                try:
                    item = unpickler.load()
                    yield item
                except:
                    break

    def get_all_data(self):
        content = []
        for item in self._gen():
            content.append(item)
        return content

    def get_label_distribution(self):
        data = self.get_all_data()
        labels = [d['label'] for d in data]
        all_labels = list(set(labels))
        count_dict = {}
        for item in all_labels:
            count_dict[item] = labels.count(item)
        return count_dict

    def play_next(self):
        x = next(self.gen)
        wav = x['wav']
        sd.play(wav, self.fs, blocking=True)


analyzer = Analyzer('assets/corpora/corpus11/unknown.p.soundcorpus.p')
count_dict = analyzer.get_label_distribution()
print(count_dict)
analyzer.play_next()

