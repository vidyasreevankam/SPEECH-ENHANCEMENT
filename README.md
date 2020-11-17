# SPEECH-ENHANCEMENT
The code here is a part of Speech-Enhancement performed used Dual Signal Transformation Using LSTM Network(DTLN).
The dataset is taken from https://datashare.is.ed.ac.uk/handle/10283/1942.
The above code is implemented only on a part of dataset in above link based on my system processor.
For better results one needs to run the above code on GPU for atleast 100 epochs to evaluate the performance of model.
Python Dependencies:
1)Tensorflow(ran on 1.13.2) 
2)librosa 
3)wavinfo 

For the conversion of sample rate you can run the file conversion.py to downsample your data. You can change the directory to the input_path files and out_path.

The training can be done by running
python run_training.py

In the run_python.py file you need to change the directories of your dataset in the respective folders. The training results are saved in the folder models_DTLN_model. The model is saved in model.h5 type and you can see the training log there.

You can change the parameters and play with the model by making modifications in the DTLN_model.py file.

For evaluation you need to run
python run_evaluation.py -i /name/of/input/folder -o /name/of/output/folder -m /name/of/the/model.h5

You can also evaluate this using pretrained models on Speech Enhancement.

In the above files models_DTLN_model you can find the trained model on small dataset comprising of 2328 sentences for clean and noisy train set and 300 sentences for clean and noisy test set ran for 10 epochs.
You can change the parameters according to your dataset and can train and evaluate the model.

In the sample_results.zip you can find the noisy and enhanced results trained on this model using small dataset for 10 epochs.
The results can be fruitful if you run on large database by changing the params.
