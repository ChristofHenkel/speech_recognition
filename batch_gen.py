import os
import pickle
import numpy as np


class BatchGen:

    def __init__(self, batch_size, soundcorpus_fp, mode = 'train'):
        self.bsize = batch_size
        self.fp = soundcorpus_fp
        #with open('assets/corpora/corpus1/nameiddict.p','rb') as f:
        #    dec, enc = pickle.load(f)
        #self.decoder = dec
        #self.encoder = enc
        self.mode = mode

    def batch_gen(self):
        x = []
        y = []
        with open(self.fp, 'rb') as file:
            unpickler = pickle.Unpickler(file)
            while True:
                #try:
                item = unpickler.load()

                #except:
                #    print('hi')
                #    x = np.asarray(x)
                #    if not self.mode in ['test']:
                #        y = np.asarray(y)
                #    # y = np.reshape(y, (self.bsize,1))
                #    yield x, y

                x.append(item['wav'])
                if self.mode in ['train','valid']:
                    y.append(item['target'])
                elif self.mode == 'test':
                    y.append(item['sample'])
                else:
                    pass
                if len(x) == self.bsize:
                    # reshape to np arrays
                    x = np.asarray(x)
                    if not self.mode in ['test']:
                        y = np.asarray(y)
                    #y = np.reshape(y, (self.bsize,1))
                    yield x, y
                    x = []
                    y = []



