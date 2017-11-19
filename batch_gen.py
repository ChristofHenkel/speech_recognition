import os
import pickle
import numpy as np


class BatchGen:

    def __init__(self, batch_size, soundcorpus_fp):
        self.bsize = batch_size
        self.num = 16000
        self.fp = soundcorpus_fp

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
                    yield x, y
                    x = []
                    y = []



