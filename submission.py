from tqdm import tqdm
import numpy as np
from glob import glob
# now we want to predict!
paths = glob(os.path.join(DATADIR, 'test/audio/*wav'))

def test_data_generator(data):
    def generator():
        for path in data:
            _, wav = wavfile.read(path)
            wav = wav.astype(np.float32) / np.iinfo(np.int16).max
            fname = os.path.basename(path)
            yield dict(
                sample=np.string_(fname),
                wav=wav,
            )

    return generator

test_input_fn = generator_input_fn(
    x=test_data_generator(paths),
    batch_size=hparams.batch_size,
    shuffle=False, 
    num_epochs=1,
    queue_capacity= 10 * hparams.batch_size,
    num_threads=1,
)

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