from DLuNET import *
from netCDF4 import Dataset
import time
import numpy as np
import matplotlib.pyplot as plt

# DATA LOAD SECTION: TODO Create a class for the dataloader
 


all_data = []
input_var_names = ['lon', 'lat',
                   'eastward_model_wind', 'northward_model_wind', 'model_speed', 'model_dir',
                   'msl', 'air_temperature', 'q', 'sst', 'uo', 'vo']
output_var_names = ['u','v']

train_input_file = "/Volumes/SSD Adrian/TFM/adrian_tfm/ASCAT_l3_collocations/2020/train/ascata_20200101_l3_asc.nc"

# TODO: Automatic load just creating the pointer to the directory
f = Dataset(train_input_file)

# Creating new variables with the proper size 1440x2880
lon = f.variables['lon'].__array__() #0-360 degrees = 2880 points
lat = f.variables['lat'].__array__() #0-180 degrees = 1440 points

lons, lats = np.meshgrid(lon, lat)

all_data.append(lons)
all_data.append(lats)

loader_input_var_names = ['eastward_model_wind', 'northward_model_wind', 'model_speed', 'model_dir', 
                          'msl', 'air_temperature', 'q', 'sst', 'uo', 'vo']

for var_name in loader_input_var_names:
    var_data = f.variables[var_name].__array__()[0]
    all_data.append(var_data)

# Actual input data to inspect: X 
input_masked_data = np.ma.MaskedArray(all_data)
# TODO: Try to add some coherency to avoid multiple dimension swap
# reshape as an image, with the channels in the last dimension
input_masked_data = np.transpose(input_masked_data,(1,2,0)) 

# 256x256 piece of image we agreed
X = input_masked_data[864:1120,2568:2824,:]

u = f.variables['eastward_wind'].__array__()[0]
v = f.variables['northward_wind'].__array__()[0]
u_model = f.variables['eastward_model_wind'].__array__()[0]
v_model = f.variables['northward_model_wind'].__array__()[0]
f.close()

# Ground truth: y 
targets = np.ma.MaskedArray([u - u_model, v - v_model])
targets = np.transpose(targets,(1,2,0))

y = targets[864:1120,2568:2824,:]




# Don't know why rows are not represented as the email (Inverted)
# Answer: Because potato

# Create a figure and plot the images side by side
plt.figure(figsize=(12, 6))

# First image
plt.subplot(1, 2, 1)  # (rows, columns, panel number)
plt.imshow(y[::-1,:,0], cmap='gray')
plt.title('ascata_20200101_l3_asc u component')
plt.axis('off')  # Optional: turn off axis

# Second image
plt.subplot(1, 2, 2)
plt.imshow(y[::-1,:,0].mask, cmap='gray')
plt.title('ascata_20200101_l3_asc u component mask')
plt.axis('off')  # Optional: turn off axis

# Display the figure
plt.show()

# UNET TRAIN SECTION

model = UNet(in_channel=12,out_channel=2)

'''
# Example dataset
input_images = torch.randn(100, 12, 256, 256)  # 100 samples, 12 channels
target_images = torch.randn(100, 2, 256, 256)  # 100 samples, 2 channels

dataset = CustomDataset(input_images, target_images)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
'''
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




'''
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