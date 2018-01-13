# speech_recognition

# Introduction

As part of kaggle competition (https://www.kaggle.com/c/tensorflow-speech-recognition-challenge)


# Steps

1. Download training and test data
2. run data_augmentation to generate background noise
3. create soundcorpora containing raw wav content and labels
4. start training
5. populate submission.csv



## download training and test data

### Training Data
https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/download/train.7z

### Test Data
https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/download/test.7z

### Sample Submission
https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/download/sample_submission.7z

### unzip the selected data
There is a kwnon issue on OSX. Use Keka www.kekaosx.com or cmd-line using brew:

`$ brew update`

`$ brew install p7zip`

`$ 7z x train.7z`

## run data_augmentation
Since all training ans test wav files have length of 1 second, we also want to split the long wav files containing background noise
into 1 second snippets.
Run `data_augmentation.py` to generate a folder containing those snippets.


## create sound corpora
Since the structure of training, validation, test, background noise etc. is quite sparse we capture all relevant information in
seperate soundcorpus files containing the wav as numpy arrays and the according labels. Soundcorpora are saved in a way that we can stream
data from them for training and on-the-fly noise mixing.

Running `create_soundcorpus.py will create different soundcorpora and an info_dictonary containing the length of each
soundcorpus. We create corpora for:
*   train
*   valid
*   test
*   background
*   unknown
*   silence

## start training `
with specific batch composition, model architecture and model hparams

### important classes and scripts

####   BatchGen



####  Model
class capturing the baseline of the model architecture from input to logits.

### Visualization

#### Confusion Matrix

#### Tensorbaord







## submission
submission.py
loads model
loads test soundcorpus
creates submission file

## extras
analyze_corpus.py

## literature

* Good MFCC explanation: http://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
* Tensorflow tutorial on speech recognition: https://www.tensorflow.org/versions/master/tutorials/audio_recognition
* Batch Normalization
* Deep Speech

## what we learned so far

* mfcc good for training commands, but overfits instantly on background noise
* since silence ( meaning non-speech sound ) is quite different from the other labels, it might be best to handle it seperately.
* Batch Normalization not good if test data hast different distribution from train data, how ever helps to escape local minima

## TODOS

* fix seed

* try with 40 nfilts
* try with one more layer lstm
* try more hidden units lstm
* one more layer conv

* unknown increase rate
* augment unknown labels
* stable seed
* submission + probs

* copy own_test corpus from Downloads
* cm at acc
* overlap and rolling at train data
* load new probs to K80-3

* t3-model17_e29 very good in all but silence and unknown
* download t_model17_e27

* models/silence_detection/model_mfcc_bsize64_e1_s105.ckpt [[409   2][  3  38]]
* models/silence_detection/model_mfcc_bsize64_e1_s70.ckpt [[410   1][  4  37]]

17
33

Distribution

yes 0.08
on = 0.08
unknown 0.09
silence 0.09

t_model11_e48_submission.csv scores 0.08 on yes
t_model11_e48_submission.csv scores 0.07 on unknown
t_model11_e48_submission.csv scores 0.04 on silence
t_model11_e48_submission.csv scores 0.07 on on