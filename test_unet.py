from DLuNET import *
from netCDF4 import Dataset as netDataset
import time
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import copy
import pickle
import wandb

'''
# Loading the data stats in the disk. Actually not neccessary, we predict gt - model differences
directory = '/Users/adrianrrrrr/Documents/TFM/adrian_tfm/ASCAT_l3_collocations/2020/stats'
with open(directory+'/variables.pkl','rb') as file:
    var_stats, gt_stats = pickle.load(file)
'''

# Loading the test data
dir_path = "/Users/adrianrrrrr/Documents/TFM/adrian_tfm/ASCAT_l3_collocations/2019/test"
input_data, ground_truth, lon, lat = MyDataLoader2(dir_path)

input_data_norm = copy.deepcopy(input_data)
ground_truth_norm = copy.deepcopy(ground_truth)

MyNorm(input_data_norm)
MyNorm(ground_truth_norm)

# Conversion numpy.ma to a standard ndarray with 0 filled values
in_data = []
for np_masked_day in input_data_norm:
    conv2ndarray = np_masked_day.filled(fill_value=0)
    conv2tensor = torch.from_numpy(conv2ndarray)
    if (mydevice == torch.device('mps')): # Double do not work properly with MPS
        conv2tensor = conv2tensor.type(torch.FloatTensor)
    in_data.append(conv2tensor)

gt_data = []
for np_masked_day in ground_truth_norm:
    conv2ndarray = np_masked_day.filled(fill_value=0)
    conv2tensor = torch.from_numpy(conv2ndarray)
    if (mydevice == torch.device('mps')): # Double do not work properly with MPS
        conv2tensor = conv2tensor.type(torch.FloatTensor)
    gt_data.append(conv2tensor)

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="TFM-UNET",

    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.001,
    "architecture": "UNET",
    "dataset": "ASCAT-B",
    "epochs": 1000,
    }
)

# Loading the best trained model
model_path = '/Users/adrianrrrrr/Documents/TFM/adrian_tfm/ASCAT_l3_collocations/2020/saved_models'
model_name = '/model_87_0DOT16214.pt' # Best model name model_<BESTEPOCH>_<BESTLOSS>.pt
model = UNet() # Model initialization
model = model.to(mydevice) # To GPU if available
model.load_state_dict(torch.load(model_path+model_name)) # Load the trained params from train_unet.py best model 
criterion = nn.MSELoss() # Loss function for regression -> MSE. No reduction (Neccessary for masking values)


# Testing loop
# We still with only one example TODO: VALIDATE WITH ALL THE EXAMPLES

best_loss = 100000 # Initialisation of best loss for saving the best model
debug = 1
model.eval() # Setting the model to eval mode 
with torch.no_grad():
    # From now batch size is 1 (One image injected to UNET, then loss is computed)
    for image, target in zip(in_data,gt_data):
        print("Processing image nr, ",debug)
        debug=debug+1
        # The UNET takes an input of shape (B,12,1440,2880) where B is directly the batch size
        # B is the number of examples processed by the model, the UNET, before updating parameters
        # Now B = 1. For B = 4, input should be of shape (4,12,l,w)
        input = image[None,:,864:1120,2568:2824].to(mydevice) # 256x256 patch + adding first dummy dimension for the UNET
        output = model(input) 
        groundt = target[None,:,864:1120,2568:2824].to(mydevice)

        # Create the mask for ignoring the zero values in the targets
        mask = groundt != 0
        loss = criterion(output,groundt)
        masked_loss = loss * mask # Apply the mask
        final_loss = masked_loss.sum() / mask.sum() # Normalize by the number of non-zero elements

        # Log the metric into wandb
        wandb.log({"loss": final_loss})

        aux = final_loss
        if aux < best_loss:
            best_loss = aux

        print(f'Loss: {final_loss:.4f}')
   

print("Testing complete!")
print("Best loss was: ","{:.6f}".format(best_loss),"loss")

wandb.finish()



