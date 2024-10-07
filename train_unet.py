from DLuNET import *
from netCDF4 import Dataset as netDataset
import time
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import copy
import pickle
import wandb

FilesNr = 9 # Number of days we get to build the batch in the training dataset. 
dir_path = "/Users/adrianrrrrr/Documents/TFM/adrian_tfm/ASCAT_l3_collocations/2020/train"
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

''' NOT WORKING AS EXPECTED
# Getting the gt mask for filtering the loss function and so
gt_mask = torch.tensor(ground_truth_norm[0].mask,dtype=torch.float32)
gt_mask = gt_mask[None,:,864:1120,2568:2824].to(mydevice)
'''

# Storing the data stats in the disk
directory = '/Users/adrianrrrrr/Documents/TFM/adrian_tfm/ASCAT_l3_collocations/2020/stats'
with open(directory+'/variables.pkl','wb') as file:
    pickle.dump((var_stats,gt_stats),file)

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="TFM-UNET",

    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.001,
    "architecture": "UNET",
    "dataset": "ASCAT-A",
    "epochs": 1000,
    }
)

# UNET TRAIN SECTION
# I am getting a patch of 256x256. Full image requires about 6GB of VRAM that I do not have in my Radeon
model = UNet() # Model initialization
model = model.to(mydevice) # To GPU if available
criterion = nn.MSELoss() # Loss function for regression -> MSE. No reduction (Neccessary for masking values)
optimizer = optim.Adam(model.parameters(), lr=0.001) # Optimizer initialisation

# Training loop

'''
# Let's overtrain the model using only one example
aux = []
aux.append(in_data[0])
in_data = aux
aux = []
aux.append(gt_data[0])
gt_data = aux
'''

best_loss = 1000 # Initialisation of best loss for saving the best model
best_epoch = 0
# Directory to save the best model
directory = '/Users/adrianrrrrr/Documents/TFM/adrian_tfm/ASCAT_l3_collocations/2020/saved_models'
file_name = '/model'
file_ext = '.pt'

num_epochs = 10 # Number of total passess through training dataset
for epoch in range(num_epochs):
    # From now batch size is 1 (One image injected to UNET, then parameters are updated)
    for image, target in zip(in_data,gt_data):
        # Patch-processing bucle. 288x288 path injected to the UNET
        for y_lat in range(0,5):
            for x_lon in range(0,10):
                lat_position_from = 288 * y_lat
                lat_position_to = lat_position_from + 288
                lon_position_from = 288 * x_lon
                lon_position_to = lon_position_from + 288

                '''
                in_data[0][:][lat_position_from:lat_position_to][lon_position_from:lon_position_to]
                '''
                print('(',x_lon,')','(',y_lat,')','gets lons from: ',lon_position_from,'to ',lon_position_to,'and lats from: ',lat_position_from,
                    ' to: ',lat_position_to)


                # The UNET takes an input of shape (B,12,1440,2880) where B is directly the batch size
                # B is the number of examples processed by the model, the UNET, before updating parameters
                # Now B = 1. For B = 4, input should be of shape (4,12,l,w). Cannot try due computing limitations
                input = image[None,:,lat_position_from:lat_position_to,lon_position_from:lon_position_to].to(mydevice) # 288x288 patch + adding first dummy dimension for the UNET
                output = model(input) 
                groundt = target[None,:,lat_position_from:lat_position_to,lon_position_from:lon_position_to].to(mydevice)

                # Create the mask for ignoring the zero values in the targets
                mask = groundt != 0
                loss = criterion(output,groundt)
                masked_loss = loss * mask # Apply the mask
                final_loss = masked_loss.sum() / mask.sum() # Normalize by the number of non-zero elements

                # Log the metric into wandb
                wandb.log({"loss": final_loss})
                
                # Zero gradients to prevent accumulation from multiple backward passes. 
                optimizer.zero_grad()

                # Computing the gradients of the loss wirth respect to each parameter (weight, bias) i.e. how much
                # each parameter needs to change to reduce the loss. Computed gradients are stored in .grad attribute
                # of each param tensor
                final_loss.backward()
                aux = final_loss
                if aux < best_loss:
                    best_loss = aux
                    best_epoch = epoch+1
                    model_state_dict = model.state_dict()

                # Update of the model paramters based on each gradient. It adjust parameters using Adam opt. algorithm.
                optimizer.step()


            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {final_loss:.4f}')




print("Training complete!")
print("Best epoch was: ",best_epoch," giving a","{:.6f}".format(best_loss),"loss")

# Saving the best model on disk with the agreed naming
rounded_loss = "{:.6f}".format(best_loss)
rounded_loss = rounded_loss[0]+'DOT'+rounded_loss[2:7]
file_save_name = directory+file_name+'_'+str(best_epoch)+'_'+rounded_loss+file_ext
torch.save(model_state_dict,file_save_name)


wandb.finish()
