from DLuNET import *
from netCDF4 import Dataset as netDataset
import time
import numpy as np
import matplotlib.pyplot as plt


FilesNr = 9

input_data, ground_truth = MyDataLoader()
MyNorm(input_data,FilesNr)
MyNorm(ground_truth,FilesNr)

# Adrian: After many attempts, fails and therapy, I will start implementing
# a simple 256x256x1 uNET with the Sea Surface Temperature (sst) variable. 
# btw UNET with 1 in channel and 2 out channels DO NOT produce same channels
# outputs...

# UNET TRAIN SECTION
model = UNet()
input_image = torch.randn(1, 1, 256, 256)  # Batch size of 1, 1 channel, 256x256 image
input_image = input_image.to(mydevice)
output_image = model(input_image)
print(output_image.shape)  # Should be [1, 2, 256, 256]

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