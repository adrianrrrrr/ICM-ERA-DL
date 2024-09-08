from DLuNET import *
from netCDF4 import Dataset as netDataset
import time
import numpy as np
import matplotlib.pyplot as plt
import copy
from maskedtensor import masked_tensor

FilesNr = 9

input_data, ground_truth = MyDataLoader()

input_data_norm = copy.deepcopy(input_data)

MyNorm(input_data_norm)
#MyNorm(ground_truth,FilesNr) # Pending the load + normalisation of the GT for the training

# Conversion numpy.ma to a standard ndarray

in_data = []
for np_masked_day in input_data_norm:
    conv2ndarray = np_masked_day.filled(fill_value=0)
    conv2tensor = torch.from_numpy(conv2ndarray)
    if (mydevice == torch.device('mps')): # Double do not work properly with MPS
        conv2tensor = conv2tensor.type(torch.FloatTensor)
    in_data.append(conv2tensor)


# UNET TRAIN SECTION
model = UNet() # Model initialization
model = model.to(mydevice) # To GPU if available
#input_image = torch.randn(1, 12, 256, 256)  # Dummy test  Tensor
 #y = targets[:,864:1120,2568:2824] # Do this indexing to get 256x256 test because I ran out memory on MPS
input_image = in_data[0] # First day
input_image = input_image[None,:,864:1120,2568:2824] # Adding the first dummy dimension simulating the epoch for the UNET to work + getting 256^2 patch
input_image = input_image.to(mydevice)
output_image = model(input_image) # Input shape should be (1,12,1440,2880)
print(output_image.shape)  # Should be [1, 2, 256, 256]




'''
Copilot UNET training model

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Define the U-Net model
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.upconv = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.upconv(x1)
        x3 = self.decoder(x2)
        return x3

# Custom dataset
class CustomDataset(Dataset):
    def __init__(self, images, targets):
        self.images = images
        self.targets = targets

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        target = self.targets[idx]
        return image, target

# Dummy data
images = torch.randn(9, 12, 1440, 2880)
targets = torch.randn(9, 2, 1440, 2880)

# Create dataset and dataloader
dataset = CustomDataset(images, targets)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Initialize model, loss function, and optimizer
model = UNet(in_channels=12, out_channels=2)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for images, targets in dataloader:
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training complete!")



'''







'''
# Example dataset
input_images = torch.randn(100, 12, 256, 256)  # 100 samples, 12 channels
target_images = torch.randn(100, 2, 256, 256)  # 100 samples, 2 channels

dataset = CustomDataset(input_images, target_images)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

input_masked_data = np.transpose(input_masked_data,(2,0,1)) # reshape as an image, with the channels in the last dimension
# 256x256 piece of image we agreed
X = input_masked_data[:,864:1120,2568:2824]


# Define the loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error Loss for regression
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Move the model to GPU if available
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Convert from Numpy to torch tensor
input = torch.from_numpy(X.astype(np.float32)) # convert to float32 because MPS to not work with float64
input = input.to(device)
target = torch.from_numpy(y.astype(np.float32))
target = target.to(device)

# Add a dummy first dimension to simulate the batch
input = input[None,:,:,:]
target = target[None,:,:,:]
output = model(input)

loss = criterion(output,target)

loss.backward()
optimizer.step()





# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Update running loss
        running_loss += loss.item() * inputs.size(0)

    # Calculate and print average loss for the epoch
    epoch_loss = running_loss / len(dataloader.dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

print("Training Complete")
'''