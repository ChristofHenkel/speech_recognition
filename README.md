# speech_recognition

# Introduction

As part of kaggle competition

Link to Tensorflow tutorial

#Steps
## download data

### Training Data
https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/download/train.7z

### Test Data
https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/download/test.7z

### Sample Submission
https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/download/sample_submission.7z


## unzip (issue with 7zip on mac)
## run data_augmentation
creates folder with noise data
## create sound corpora
running 'create_soundcorpus.py' will create 6 different soundcorpora and an info_dictonary

## start training with specific batch composition, model architecture and model hparams

## submission
submission.py
loads model
loads test soundcorpus
creates submission file

## extras
analyze_corpus.py

## literature

# Good MFCC explanation:
http://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html

# what we learned

mfcc good for training commands, but overfits instantly on background noise