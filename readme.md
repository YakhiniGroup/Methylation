# Predicting Methylation from Sequence and Gene Expression Using Deep Learning with Attention

This program is as described in the paper: "Predicting Methylation from Sequence and Gene Expression Using Deep Learning with Attention" by Levy-Jurgenson et al.
With this program you can:

(1) Predict methylation levels for a given sample at a given CpG site using the sample's gene-expression data, and the CpG's ambient sequence. 

(2) Train/test your own version of the model either from scratch or using one of the pre-trained models described in the paper as a staring point (Model 1, Model 2 or Model 3).

## Getting Started

Below you will find explanations, including examples, for each of the options above. 
For prediction purposes, as well as for training with one of our pre-trained models, you will need the trained weights to reside in the out folder. To obtain them, please feel free to contact us at levyalona at gmail (the files are ~2-4GB). In the near future we will have a direct download link.  

### Prerequisites 

Requires:
Python 3.5.6 (or 3.5 should do).

Tensorflow 1.4+  

## (1) Prediction

To predict methylation levels, use predict.py. Inside, you will find a designated area with a few settings that enable you to specify the input files (such as the gene-expression measurements file). 
For your convenience, we supplied sample files in the res folder for format only (to keep them small, they do not contain all the genes, but you can use them as mock training in (2) below). You should adhere to the format according to these files.  Below is an example of what you would have to supply. This is also the default in predict.py.
The load_model_ID parameter corresponds to which of the models from the paper you would like to use for the prediction (see paper for details, or if you're not sure - please contact us - we will gladly assist!) 

### Example
```
...
load_model_ID = 3
filename_sequence = 'probeToOneHotAll_sample_mini.csv'
filename_expression = 'e_sample_mini.csv'
filename_dist = 'd_sample_mini.csv'
...
```

## (2) Training your own model

To train from scratch, or from a pre-trained model, use run.py. Inside, you will find a designated area with a few settings that enable you to specify which, if any, model you want to use as a starting point for your training. 
If you want to train from scratch - use load_model_ID = 0. This is also further explained in run.py. You can also control whether you are testing or training.

Note that there is no need to split your data into train/val/test in advance - this will be done automatically and randomly when first running the model (this is controlled in dataset.py).  
For your convenience, we supplied sample files in the res folder. You should adhere to the format according to these files. To supply your file names, as well as other settings, use conf.py. 
Below is an example of what yo  u would have to supply, both in run.py and in conf.py.
The load_model_ID parameter corresponds to which of the models from the paper you would like to use as pre-training (see paper for details, or if you're not sure - please contact us - we will gladly assist!) 

### Example

run.py
```
...
load_model_ID = 0 # train from scratch
test_time = True   # when you're ready to test your model (will use the automatically, and random, test set it created when starting to train)
save_models = True # throughout training, checkpoints will be automatically saved upon improvement on validation set
...
```

conf.py
```
...
filename_sequence = 'probeToOneHotAll_sample_mini.csv'
filename_expression = 'e_sample_mini.csv'
filename_dist = 'd_sample_mini.csv'
...
validation_portion_subjects = 0.1
validation_portion_probes = 0.1
train_portion_probes = 0.7
...
```

If you need help, or have any questions, please feel free to contact us at: levyalona at gmail....

## Authors

* **Alona Levy-Jurgenson** 
* **Xavier Tekpli**
* **Vessela N. Kristensen**
* **Zohar Yakhini**

Please cite our paper if you use this project.