from DLuNET import *
from netCDF4 import Dataset as netDataset
import time
import numpy as np
import matplotlib.pyplot as plt
import copy
from maskedtensor import masked_tensor

FilesNr = 9 # Number of days we get to form the epoch, the training dataset. 

input_data, ground_truth = MyDataLoader()

input_data_norm = copy.deepcopy(input_data)
ground_truth_norm = copy.deepcopy(ground_truth)

MyNorm(input_data_norm)
MyNorm(ground_truth_norm)

# Conversion numpy.ma to a standard ndarray

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

# UNET TRAIN SECTION
# I am getting a patch of 256x256. Full image requires about 6GB of VRAM that I do not have in my Radeon
model = UNet() # Model initialization
model = model.to(mydevice) # To GPU if available
criterion = nn.MSELoss() # Loss function for regression -> MSE
optimizer = optim.Adam(model.parameters(), lr=0.001) # Optimizer initialisation

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    index = 0
    for image, target in zip(in_data,gt_data):
        input = image[None,:,864:1120,2568:2824].to(mydevice)
        output = model(input) # 256x256 patch + adding first dummy dimension for the UNET
        target = target[None,:,864:1120,2568:2824].to(mydevice)
        loss = criterion(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}') ## loss.item() is the value of MSE

print("Training complete!")

directory = '/Users/adrianrrrrr/Documents/TFM/adrian_tfm/ASCAT_l3_collocations/2020/saved_models'
file_name = '/model.pth'

torch.save(model.state_dict(),directory+file_name)

'''
#input_image = torch.randn(1, 12, 256, 256)  # Dummy test  Tensor
 #y = targets[:,864:1120,2568:2824] # Do this indexing to get 256x256 test because I ran out memory on MPS
input_image = in_data[0] # First day
input_image = input_image[None,:,864:1120,2568:2824] # Adding the first dummy dimension simulating the epoch for the UNET to work + getting 256^2 patch
input_image = input_image.to(mydevice)
output_image = model(input_image) # Input shape should be (1,12,1440,2880)
print(output_image.shape)  # Should be [1, 2, 256, 256]
'''