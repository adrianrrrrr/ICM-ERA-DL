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

'''
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
'''

# UNET TRAIN SECTION
# I am getting a patch of 256x256. Full image requires about 6GB of VRAM that I do not have in my Radeon
model = UNet() # Model initialization
model = model.to(mydevice) # To GPU if available
criterion = nn.MSELoss(reduction='none') # Loss function for regression -> MSE. No reduction (Neccessary for masking values)
learning_rate = 0.01
optimizer = optim.Adam(model.parameters(), lr=learning_rate) # Optimizer initialisation

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
running_loss_train = [] # All train losses
loss_train = [] # Epoch averaged train losses

best_epoch = 0
# Directory to save the best model
directory = '/Users/adrianrrrrr/Documents/TFM/adrian_tfm/ASCAT_l3_collocations/2020/saved_models'
file_name = '/model'
file_ext = '.pt'

# Training loop
# Loading 60% of X for training
X_train = in_data[0:18]
y_train = gt_data[0:18]
num_epochs = 50 # Number of total passess through training dataset
for epoch in range(num_epochs):

    # From now batch size is 1 (One image injected to UNET, then parameters are updated)
    for image, target in zip(X_train,y_train):

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
        #wandb.log({"loss": final_loss})
        
        # Zero gradients to prevent accumulation from multiple backward passes. 
        optimizer.zero_grad()

        # Computing the gradients of the loss wirth respect to each parameter (weight, bias) i.e. how much
        # each parameter needs to change to reduce the loss. Computed gradients are stored in .grad attribute
        # of each param tensor
        final_loss.backward()

        running_loss_train.append(float(final_loss)) # Vector with all individual losses 
        aux = final_loss
        if aux < best_loss:
            best_loss = aux
            best_epoch = epoch+1
            #model_state_dict = model.state_dict()

        # Gradient clipping to try to smooth the learning
        torch.nn.utils.clip_grad_norm_(model.parameters(),1)
        # Update of the model paramters based on each gradient. It adjust parameters using Adam opt. algorithm.
        optimizer.step()

    # Evaluar aquí (Al final de cada epoch) el modelo que se ha entrenado en un set de validación a parte
    # Aquí tendría que hacer la parte del script que tengo abajo y calcular el error media.
    # Poner una variable auxiliar, guardar el error de validación en ella y cuando suba es donde paramos y guardamos el modelo
    # (Anterior)
    print(f'Epoch [{epoch+1}/{num_epochs}], Training loss: {final_loss:.4f}')

print("Training complete")
print("Best epoch was: ",best_epoch," giving a best running loss of ","{:.6f}".format(best_loss))

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,6))

ax1.plot(running_loss_train,label='UNet training running loss')
ax1.set_title('Train loss [Running loss]')
ax1.set_xlabel('# image')
ax1.set_ylabel('Loss')
max_y = max(running_loss_train) + 1
ax1.set_ylim(0,max_y)
ax1.set_facecolor('lightgrey')
plot_text = f'Learning rate = {learning_rate}'
props = dict(boxstyle='round',facecolor='white',alpha=0.5)
ax1.text(0.7,0.9,plot_text,transform=ax1.transAxes,fontsize=9,verticalalignment='top', bbox=props)

epoch_size = len(X_train)
epoch_avg_train = moving_average(running_loss_train,epoch_size)
ax2.plot(epoch_avg_train,label='UNet training loss')
ax2.set_title('Train loss [Epoch averaged]')
ax2.set_xlabel('# Epoch')
ax2.set_ylabel('Loss')
ax2.set_ylim(0,max_y)
ax2.set_facecolor('lightgrey')
plot_text = f'Learning rate = {learning_rate}'
props = dict(boxstyle='round',facecolor='white',alpha=0.5)
ax2.text(0.7,0.9,plot_text,transform=ax2.transAxes,fontsize=9,verticalalignment='top', bbox=props)

fig_dir = '/Users/adrianrrrrr/Documents/TFM/adrian_tfm/ASCAT_l3_collocations/2020'
str_lr = str(learning_rate)
save_name = fig_dir+'/train_lr_'+str_lr[0]+'DOT'+str_lr[2:]+'.png'
plt.savefig(save_name)
plt.show()

# Testing loop
losses_test = []
model.eval()
X_test = in_data[18:31]
y_test = gt_data[18:31]
with torch.no_grad():
    for image, target in zip(X_test,y_test):

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

        losses_test.append(float(final_loss)) # Vector of losses for debugging
        aux = final_loss
        if aux < best_loss:
            best_loss = aux

avg_test_loss = sum(losses_test) / len(losses_test)

print(f'Avg testing loss = {avg_test_loss:.3f} with learning rate = {learning_rate}')
print(f'Best running testing loss = {best_loss:.3f} with learning rate = {learning_rate}')

fig, ax = plt.subplots()
x = [i for i in range(1,len(losses_test)+1)]
ax.plot(x,losses_test,label='UNet testing loss')
plt.title('Testing loss')
plt.xlabel('# Day')
plt.ylabel('Loss')
plt.ylim(0,5)
ax.set_facecolor('lightgrey')
plot_text = f'Learning rate = {learning_rate}'
plot_text2 = f'Avg test loss = {avg_test_loss:.3f}'
props = dict(boxstyle='round',facecolor='white',alpha=0.5)
ax.text(0.7,0.97,plot_text,transform=ax.transAxes,fontsize=9,verticalalignment='top', bbox=props)
ax.text(0.7,0.87,plot_text2,transform=ax.transAxes,fontsize=9,verticalalignment='top', bbox=props)


fig_dir = '/Users/adrianrrrrr/Documents/TFM/adrian_tfm/ASCAT_l3_collocations/2020'
str_lr = str(learning_rate)
save_name = fig_dir+'/test_lr_'+str_lr[0]+'DOT'+str_lr[2:]+'.png'
plt.savefig(save_name)
plt.show()


# Saving the best model on disk with the agreed naming
model_state_dict = model.state_dict()
rounded_loss = "{:.6f}".format(avg_test_loss) #best_loss is the best testing loss
rounded_loss = rounded_loss[0]+'DOT'+rounded_loss[2:7]
file_save_name = directory+file_name+'_'+str(best_epoch)+'_'+rounded_loss+file_ext
torch.save(model_state_dict,file_save_name)

#wandb.finish()
