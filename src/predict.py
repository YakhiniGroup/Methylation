import random
from src.conf import Conf, ConfSample
import src.dataset as Dataset
from src.model import Model
import pandas as pd
import tensorflow as tf

'''
Used to obtain methylation predictions per cpg and sample (if you are looking to train a new model, or use an existing one for
transfer learning, use run.py). 
Modify the settings below according to your needs:
 (1) load_model = 1/2/3 if your CpG is within a 2K of one of the genes in the run_example file, choose 1, if within 10K choose 2 etc. according to paper. If
you're not sure - either contact us (alonal at cs.technion.ac.il) or simply use the most general model - 3). 
As the weights are large files (~1G) please contact us and we will send them directly to you. In the near future they 
will be available directly for download.
 (2) Provide the names of your files, which should be placed under the res directory. You will see there a sample 
     of the required files and their format. Note that you only need 3 out of the 4 (probeToOneHot, e and d).
The output will be saved in the out dir when done. You might want to run on a handful of examples before running on many,
just to verify the output.
In case you encounter any issues - feel free to contact us at alonal at cs.technion.ac.il or levyalona at gmail. 
'''

####  YOUR SETTINGS - START ####

load_model_ID = 2
filename_sequence = 'probeToOneHotAll_sample_mini.csv'
filename_expression = 'e_sample_mini.csv'
filename_dist = 'd_sample_mini.csv'
run_example = False  # use this if you want to see an example run (will use the filenames in ConfSample, as seen below, which are also the default above)

####  YOUR SETTINGS - END ####


if run_example:
    Conf = ConfSample
    filename_sequence = ConfSample.filename_sequence
    filename_expression = ConfSample.filename_expression
    filename_dist = ConfSample.filename_dist


train, validation, test, validation_ch3_blind, test_ch3_blind, validation_e_blind, test_e_blind = Dataset.read_data_sets(
    filename_sequence=filename_sequence,
    filename_expression=filename_expression,
    filename_labels=None,
    filename_dist=filename_dist,
    train_portion_subjects=0,
    train_portion_probes=0, validation_portion_subjects=0,
    validation_portion_probes=0, directory='../res/', is_prediction=True)

# train, validation, test, validation_ch3_blind, test_ch3_blind, validation_e_blind, test_e_blind = None, None, None, None, None, None, None

d = pd.read_csv('../res/' + filename_expression, nrows=1)
n_genes = len(d.columns)-1


ff_hidden_units = [[50,0]]
ff_n_hidden = 3
conv_filters = [64]
conv_pools = [10]
conv_strides = [10]
connected_hidden_units = [[50,0]]
connected_n_hidden = 3
reg_scales = [0.0]
optimizers = [tf.train.AdamOptimizer]
losses = [tf.losses.absolute_difference]
model_name_suffix = ''


regularization_scale = reg_scales[0]
multiplyNumUnitsBy = 1
numLayers = 0
neighborAlpha = 0
c = "all"
lr = 0
n_quant = 0
ff_h = ff_hidden_units[0]
conv_f = conv_filters[0]
conv_p = conv_pools[0]
conv_s = conv_strides[0]
connected_h = connected_hidden_units[0]

modelID = str(lr) + "_" + str(ff_h[0]) +"_"+str(conv_f)+"_"+str(conv_p)+"_"+str(connected_h)+"_"+str(random.randint(0,1000))\
          +"_"+str(optimizers[0]).split(".")[-2]+"_"+str(losses[0]).split(" ")[1]+model_name_suffix
model = Model(run_example, modelID, lr, multiplyNumUnitsBy, n_quant, neighborAlpha, c,
              ff_h[0], ff_h[1], conv_f, conv_p, conv_s, connected_h[0], connected_h[1], regularization_scale,
              train, validation, test, validation_ch3_blind, test_ch3_blind, validation_e_blind, test_e_blind, n_genes,
              ff_n_hidden, connected_n_hidden, optimizers[0], losses[0], load_model=load_model_ID, is_prediction=True, test_time=True, save_models=False, save_weights=False)
model.build(with_autoencoder=False)


