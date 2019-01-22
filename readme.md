# Predicting Methylation from Sequence and Gene Expression Using Deep Learning with Attention

This program is as described in the paper: "Predicting Methylation from Sequence and Gene Expression Using Deep Learning with Attention" by Levy-Jurgenson et al.
With this program you can:

(1) Predict methylation levels for a given sample at a given CpG site using the sample's gene-expression data, and the CpG's ambient sequence. 

(2) Train/test your own version of the model either from scratch or using one of the pre-trained models described in the paper as a staring point (Model 1, Model 2 or Model 3).

## Getting Started 

Below you will find explanations, including examples, for each of the options above. 
For prediction purposes, as well as for training with one of our pre-trained models, you will need the trained weights to reside in the out/ch3_e_blind folder (after unzipping, place all ".ckpt" files **directly** under this folder, e.g. out/ch3_e_blind/model_1.ckpt... etc.). You can download them directly from [this link](https://drive.google.com/open?id=17UGNCgK6yfiZJeHJrDOolgFa2rnz4ICN) (note that the files are ~2-4GB).   

### Prerequisites 

Requires:

Python 3.5.6 (or 3.5 should do). 

Pandas (pip3 install pandas)

sklearn (pip3 install scikit-learn)

Tensorflow 1.4+ (pip3 install tensorflow)


## (1) Prediction

To predict methylation levels, use predict.py. Inside, you will find a designated area with a few settings that enable you to specify the input files (such as the gene-expression measurements file). 
For your convenience, we supplied sample files in the res folder (NOTE - these files contain little data and are for demonstration purposes only). You should adhere to the format according to these files. Below is an example of the settings you would have to modify. This is also the default in predict.py.
The load_model_ID parameter corresponds to which of the models from the paper you would like to use for the prediction (see paper for details, or if you're not sure - you're welcome to contact us) 
Note that you can use preprocessor.py and distances.py to help you prepare your data.

Before running:

* If running in terminal, the project assumes that you are running prediction.py from within the src folder (all paths are relative to src), so cd to src first.
* If using an interpreter, you may need to mark the src directory as Sources Root (on PyCharm: right-click src folder -> Mark Directory As -> Source Root) or add "src." to the local import statements (e.g. "import conf" becomes "import src.conf").
* Make sure to have an out directory that contains a sub-folder named: "ch3_e_blind" in which you should place the model's weights (after unzipping, place all ".ckpt" files **directly** under this folder, e.g. out/ch3_e_blind/model_1.ckpt...  etc.). You can download the weights directly from [this link](https://drive.google.com/open?id=17UGNCgK6yfiZJeHJrDOolgFa2rnz4ICN) (note that the files are ~2-4GB). 


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
For your convenience, we supplied sample files in the res folder for format only (to keep them small, they do not contain many cpgs or samples, so don't expect this to train very well). You should adhere to the format according to these files. To supply your file names, as well as other settings, use conf.py. 
Below is an example of what you would have to supply, both in run.py and in conf.py.
The load_model_ID parameter corresponds to which of the models from the paper you would like to use as pre-training (see paper for details, or if you're not sure - you're welcome to contact us) 
Note that you can use preprocessor.py and distances.py to help you prepare your data.

Before running:

* If running in terminal, the project assumes that you are running run.py from within the src folder (all paths are relative to src), so cd to src first.
* If using an interpreter, you may need to mark the src directory as Sources Root (on PyCharm: right-click src folder -> Mark Directory As -> Source Root) or add "src." to the local import statements (e.g. "import conf" becomes "import src.conf").
* Unless you are running the default sample run (load_model_ID = 0), you need to make sure that the following directories are in place:
    * An out directory that contains the sub-folders: "ch3_e_blind", "ch3_blind", "e_blind", "plots", "postTrainingAnalysis". If using a pre-trained model, its .ckpt files should be placed under:  ch3_e_blind (after unzipping, place all ".ckpt" files **directly** under this folder, e.g. out/ch3_e_blind/model_1.ckpt...  etc.). You can download the weights directly from [this link](https://drive.google.com/open?id=17UGNCgK6yfiZJeHJrDOolgFa2rnz4ICN) (note that the files are ~2-4GB). 
    * A logs directory.



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
