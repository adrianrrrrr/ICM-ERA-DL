from preprocessing import *


# We are going to perform a simple Exploratory Data Analysis on the ASCAT data. Very urgent to
# do so in order to align all of us with having clear the current problems to work on

import time
import numpy as np
import matplotlib.pyplot as plt

start = time.time()

def generate_inputs_train(xgboost_params, input_var_names, train_input_dir, eval_input_dir, np_train_files_dir, np_eval_files_dir, prefix):
    #xgb = XGBRegressor(**xgboost_params)
    # TODO PSEUDOCODE: initialize the uNET with the default constructor

    # Full model
    xbg_train_generator = XGBoostInputGenerator(input_var_names, target_var_names, input_dir=train_input_dir,
                                                output_dir=np_train_files_dir, downsample_ratio=0.1,
                                                records_per_file=1e10,
                                                seed=123, out_file_prefix=prefix)
    train_np_flist = xbg_train_generator.generate_np_files()
    xbg_eval_generator = XGBoostInputGenerator(input_var_names, target_var_names, input_dir=eval_input_dir,
                                               output_dir=np_eval_files_dir, downsample_ratio=0.1,
                                               records_per_file=1e10,
                                               seed=123, out_file_prefix=prefix)
    eval_np_flist = xbg_eval_generator.generate_np_files()
    train_data = np.load(train_np_flist[0])
    eval_data = np.load(eval_np_flist[0])

def generate_inputs_train2(uNET_params, input_var_names, train_input_dir, eval_input_dir, np_train_files_dir, np_eval_files_dir, prefix):
    # TODO PSEUDOCODE: initialize the uNET with the default constructor

    # Full model
    uNET_train_generator = uNETInputGenerator(input_var_names, target_var_names, input_dir=train_input_dir,
                                                output_dir=np_train_files_dir, downsample_ratio=0.1,
                                                records_per_file=1e10,
                                                seed=123, out_file_prefix=prefix)
    train_np_flist = uNET_train_generator.generate_np_files()
    uNET_eval_generator = uNETInputGenerator(input_var_names, target_var_names, input_dir=eval_input_dir,
                                                output_dir=np_eval_files_dir, downsample_ratio=0.1,
                                                records_per_file=1e10,
                                                seed=123, out_file_prefix=prefix)
    eval_np_flist = uNET_eval_generator.generate_np_files()
    train_data = np.load(train_np_flist[0])
    eval_data = np.load(eval_np_flist[0])


# TODO PSEUDOCODE: From here it is used the XGBRegressor from xgboost! I write in pseudocode for the uNET

    # Forward each image in train_data['inputs'] then compare with the ground truth stored in train_data['targets'],
    # also indicate the training set for computing the metrics and so
'''
    xgb.fit(train_data['inputs'], train_data['targets'], eval_set=[(eval_data['inputs'], eval_data['targets'])],
            verbose=True)
    evals_metrics = xgb.evals_result()
    model_path = f"{save_model_folder}{prefix}1000.json"
    xgb.save_model(model_path)


    return xgb, evals_metrics, model_path
'''
input_var_names = ['lon', 'lat', 'eastward_model_wind', 'northward_model_wind', 'model_speed', 'model_dir',
                   'msl', 'air_temperature', 'q', 'sst', 'uo', 'vo']
print("Number of input variables / features = ",len(input_var_names))

scat_model_var_names = {'scat': ['eastward_wind', 'northward_wind'],
                        'model':['eastward_model_wind', 'northward_model_wind']}
target_var_names = ['u_diff', 'v_diff']
print("Number of output targets = ",len(target_var_names))

# FROM HERE WE START TO LOAD & ADAPT THE DATA WITH THE GIVEN HELPER FUNCTIONS. BUT JUST FOR EDA FROM NOW

#File path env setup. Currently local execution for Adrian's iMac
# Train period from 02/01/2020 - 06/03/2020 both included
train_input_dir = "/Volumes/SSD Adrian/TFM/adrian_tfm/ASCAT_l3_collocations/2020/train/"
# Test period from  10/03/2020 - 01/05/2020 both included
eval_input_dir = "/Volumes/SSD Adrian/TFM/adrian_tfm/ASCAT_l3_collocations/2020/test/"

np_train_files_dir = "/Volumes/SSD Adrian/TFM/adrian_tfm/ASCAT_l3_collocations/2020/uNET_np_data/train/"
np_eval_files_dir = "/Volumes/SSD Adrian/TFM/adrian_tfm/ASCAT_l3_collocations/2020/uNET_np_data/test/"

plots_folder = "/Volumes/SSD Adrian/TFM/adrian_tfm/ASCAT_l3_collocations/2020/plots/uNET_importance/"
save_model_folder = "/Volumes/SSD Adrian/TFM/adrian_tfm/ASCAT_l3_collocations/2020/saved_models/"
file_prefix = "allvars_cpu_"

# Let's define some hyper-parameters
hparams = {
    'batch_size': 64,
    'num_epochs': 10,
    'test_batch_size': 64,
    'learning_rate': 1e-3,
    'log_interval': 100,
}

# Condition to generate the data only one time. A improvement may be to try to read folder and only generate
# when folders do not exist
if True:
    generate_inputs_train2(hparams, input_var_names, train_input_dir, eval_input_dir, np_train_files_dir, 
                        np_eval_files_dir, prefix=file_prefix)
    
#xgb = XGBRegressor(**xgboost_params) | Not yet
train_fn = np_train_files_dir + "allvars_cpu_000.npz"
val_fn = np_eval_files_dir + "allvars_cpu_000.npz"

train_data = np.load(train_fn)
eval_data = np.load(val_fn)

# Now I can play freely with the data. It is from here when I am free to write the any code I want