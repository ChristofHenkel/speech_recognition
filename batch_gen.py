import os
import pickle
import numpy as np


class SoundCorpus:

    def __init__(self, soundcorpus_dir, mode = 'train'):
        self.mode = mode
        self.fp = soundcorpus_dir + [fn for fn in os.listdir(soundcorpus_dir) if fn.startswith(self.mode)][0]
        self.info_dict_fp = soundcorpus_dir + 'infos.p'
        self.decoder = None
        self.encoder = None
        self.len = None
        self._load_info_dict()


    def _load_info_dict(self):
        with open(self.info_dict_fp,'rb') as f:
            content = pickle.load(f)
            self.decoder = content['id2name']
            self.encoder = content['name2id']
            self.len = content['len_' + self.mode]

    def batch_gen(self,batch_size):
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
                if len(x) == batch_size:
                    # reshape to np arrays
                    x = np.asarray(x)
                    if not self.mode in ['test']:
                        y = np.asarray(y)
                    #y = np.reshape(y, (self.bsize,1))
                    yield x, y
                    x = []
                    y = []



