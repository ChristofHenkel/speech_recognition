import pickle
from collections import Counter

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

corpus = CorpusGen('assets/corpora/corpus7/train.pm.soundcorpus.p')
counter = Counter()

data = []
for item in corpus.gen():
    data.append(item)

labels = [d['target'] for d in data]
all_labels = list(set(labels))
count_dict = {}
for item in all_labels:
    count_dict[item] = labels.count(item)
print(count_dict)

# analyse training in classification per label