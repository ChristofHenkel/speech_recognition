import os
import pickle
import numpy as np


class BatchGen:

    def __init__(self, batch_size, soundcorpus_fp):
        self.bsize = batch_size
        self.num = 16000
        self.fp = soundcorpus_fp
        with open('assets/corpora/corpus1/nameiddict.p','rb') as f:
            dec, enc = pickle.load(f)
        self.decoder = dec
        self.encoder = enc

    def batch_gen(self):
        x = []
        y = []
        with open(self.fp, 'rb') as file:
            unpickler = pickle.Unpickler(file)
            while True:
                item = unpickler.load()
                x.append(item['wav'])
                y.append(item['target'])
                if len(x) == self.bsize:
                    # reshape to np arrays
                    x = np.asarray(x)
                    y = np.asarray(y)
                    y = np.reshape(y, (self.bsize,1))
                    yield x, y
                    x = []
                    y = []



