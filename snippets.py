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

r1 = [int(v) for v in ['258', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']]
r2 = [7  , 6 , 26,  94,   7,  49,   1,  15,  40,   2,   0,  11]
r3 = [10 ,  1, 107,  80 , 13 , 22 ,  0 , 13 , 10 ,  1 ,  0 ,  4]
r4 = [1  , 3 , 16 ,163  , 6 , 48 ,  0  , 5  ,10  , 1  , 0 , 17]
r5 =[ 15  , 1  ,17 ,114 , 55 , 13 ,  0 ,  9 , 22 ,  5 ,  0 ,  9]
r6 =[  1  , 1  , 6 , 97 ,  3 , 87  , 1 , 12 , 46  , 0 ,  0 , 10]
r7 =[  8  , 6 , 86  ,84 , 13 , 24 ,  1 ,  9 ,  9 ,  1 ,  0 ,  6]
r8 =[  9  , 3  ,32 ,112  , 9 , 26  , 1 , 36 , 19 ,  0 ,  0  , 9]
r9 =[  8  , 2 , 12,  94  , 9 , 52 ,  0  , 6  ,72  , 0 ,  0 ,  2]
r10 = [ 16  , 1  ,39  ,74  ,29 , 42 ,  0  , 6  ,37 ,  9 ,  0 ,  3]
r11 =[ 15  , 6 , 17 , 71,  50 , 37 ,  0 ,  6 , 32 ,  2 ,  1,   9]
r12 =[ 11 ,  1 ,  6, 151 ,  5 , 42 ,  0  , 8 , 16 ,  0 ,  0 , 20]

cm = np.asarray([r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12])
sums = []
for k in range(12):
    s = sum(cm[k,:])
    print(s)
    sums.append(s)
