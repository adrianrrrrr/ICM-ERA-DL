import torch
import torch.nn as nn
import torch.nn.functional as F


for y in range(1,6):
    for x in range(1,11):
        if (x == 2) and (y == 4):
            # No printee nada
            # print('NANAI')
            continue
        else:
            print('(',x,',',y,')')


'''
file_path = '/home/usuaris/imatge/adrian.ramos/saving_file.txt'
if torch.cuda.is_available():
    print("Cuda is available. There are ",torch.cuda.device_count()," devices")
    print("Current device is ",torch.cuda.current_device()," named: ",torch.cuda.get_device_name(0))
# Cuda is available. There are  1  devices
# Current device is  0  named:  NVIDIA GeForce RTX 2080 Ti

else:
    print("Fail")
# Bucle for patch processing
for y_lat in range(0,5):
    for x_lon in range(0,10):
        lat_position_from = 288 * y_lat
        lat_position_to = lat_position_from + 288
        lon_position_from = 288 * x_lon
        lon_position_to = lon_position_from + 288
        in_data[0][:][lat_position_from:lat_position_to][lon_position_from:lon_position_to]

        print('(',x_lon,')','(',y_lat,')','gets lons from: ',lon_position_from,'to ',lon_position_to,'and lats from: ',lat_position_from,
              ' to: ',lat_position_to)

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