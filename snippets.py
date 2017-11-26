from batch_gen import SoundCorpus
import numpy as np
from glob import glob
import os

from shutil import copyfile


paths = glob(os.path.join('assets', 'test/audio/*wav'))
np.random.shuffle(paths)

p = paths[:1000]

for fp in p:
    fn = fp.split('/')[3]
    copyfile(fp, 'assets/new_test/clips/' + fn)



