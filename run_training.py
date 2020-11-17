#!/usr/bin/env python3
"""
Script to train the DTLN model in default settings. The folders for noisy and
clean files are expected to have the same number of files and the files to 
have the same name. The training procedure always saves the best weights of 
the model into the folder "./models_'runName'/". Also a log file of the 
training progress is written there. To change any parameters go to the 
"DTLN_model.py" file.If you use a custom data set, just play around with
the parameters.

Please change the folder names before starting the training. 

Example call:
    $python run_training.py
"""

from DTLN_model import DTLN_model
import os

# use the GPU with idx 0
os.environ["CUDA_VISIBLE_DEVICES"]='0'
# activate this for some reproducibility
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# path to folder containing the noisy or mixed audio training files
path_to_train_mix = '/home/apiiit-rkv/Desktop/sarathi/DTLN-master/data/8K/noisy_trainset_wav/'
# path to folder containing the clean/speech files for training
path_to_train_speech = '/home/apiiit-rkv/Desktop/sarathi/DTLN-master/data/8K/clean_trainset_wav/'
# path to folder containing the noisy or mixed audio validation data
path_to_val_mix = '/home/apiiit-rkv/Desktop/sarathi/DTLN-master/data/8K/noisy_testset_wav/'
# path to folder containing the clean audio validation data
path_to_val_speech = '/home/apiiit-rkv/Desktop/sarathi/DTLN-master/data/8K/clean_testset_wav/'

# name your training run
runName = 'DTLN_model'
# create instance of the DTLN model class
modelTrainer = DTLN_model()
# build the model
modelTrainer.build_DTLN_model()
# compile it with optimizer and cost function for training
modelTrainer.compile_model()
# train the model
modelTrainer.train_model(runName, path_to_train_mix, path_to_train_speech,
                         path_to_val_mix, path_to_val_speech)



