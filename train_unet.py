from DLuNET import *
from netCDF4 import Dataset as netDataset
import time
import numpy as np
import matplotlib.pyplot as plt
import copy
import pickle

FilesNr = 9 # Number of days we get to build the batch in the training dataset. 

input_data, ground_truth = MyDataLoader()

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

# UNET TRAIN SECTION
# I am getting a patch of 256x256. Full image requires about 6GB of VRAM that I do not have in my Radeon
model = UNet() # Model initialization
model = model.to(mydevice) # To GPU if available
criterion = nn.MSELoss() # Loss function for regression -> MSE. No reduction (Neccessary for masking values)
optimizer = optim.Adam(model.parameters(), lr=0.001) # Optimizer initialisation

# Training loop
# Let's overtrain the model using only one example
aux = []
aux.append(in_data[0])
in_data = aux
aux = []
aux.append(gt_data[0])
gt_data = aux

best_loss = 1000 # Initialisation of best loss for saving the best model
best_epoch = 0
# Directory to save the best model
directory = '/Users/adrianrrrrr/Documents/TFM/adrian_tfm/ASCAT_l3_collocations/2020/saved_models'
file_name = '/model'
file_ext = '.pt'

num_epochs = 20 # Number of total passess through training dataset
for epoch in range(num_epochs):

    # From now batch size is 1 (One image injected to UNET, then parameters are updated)
    for image, target in zip(in_data,gt_data):

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

# Ploting the output of the model to see the result of prediction 

# Saving the best model on disk with the agreed naming
rounded_loss = "{:.6f}".format(best_loss)
rounded_loss = rounded_loss[0]+'DOT'+rounded_loss[2:7]
file_save_name = directory+file_name+'_'+str(best_epoch)+'_'+rounded_loss+file_ext
torch.save(model_state_dict,file_save_name)

'''
# Loading the best model, making a prediction and ploting 
# Loading the best trained model
model_path = '/Users/adrianrrrrr/Documents/TFM/adrian_tfm/ASCAT_l3_collocations/2020/saved_models'
model_name = '/model_20_0DOT17350.pt'
model = UNet() # Model initialization
model = model.to(mydevice) # To GPU if available
model.load_state_dict(torch.load(model_path+model_name))

output = model(input)
out_image = output[0].to(torch.device('cpu'))
out_image = out_image.detach().numpy()
out_image = np.transpose(out_image,(1,2,0))

combined_image = np.hstack((out_image[::-1,:,0],out_image[::-1,:,1]))
plt.imshow(combined_image,vmin=-0.5,vmax=0.5,cmap='bwr')
plt.title('u and v components of the best prediction in train')
'''