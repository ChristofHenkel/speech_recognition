from tqdm import tqdm
import numpy as np
from glob import glob
import os
# now we want to predict!



model = create_model(config=run_config, hparams=hparams)
it = model.predict(input_fn=test_input_fn)


# last batch will contain padding, so remove duplicates
submission = dict()
for t in tqdm(it):
    fname, label = t['sample'].decode(), id2name[t['label']]
    submission[fname] = label

with open(os.path.join(model_dir, 'submission.csv'), 'w') as fout:
    fout.write('fname,label\n')
    for fname, label in submission.items():
        fout.write('{},{}\n'.format(fname, label))