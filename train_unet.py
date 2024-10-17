from DLuNET import *
from netCDF4 import Dataset as netDataset
import time
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import copy
import pickle
import wandb

dir_path = "/Users/adrianrrrrr/Documents/TFM/adrian_tfm/ASCAT_l3_collocations/2020/train"
num_days = 31
input_data, ground_truth, lon, lat = MyDataLoader2(dir_path,num_days)

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
learning_rate = 0.001
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

# Train eval variables
best_train_loss = 1000 # Initialisation of best loss for saving the best model
loss_train = [] # Epoch averaged train losses
best_epoch = 0
# 60% of data for training
X_train = in_data[0:18]
y_train = gt_data[0:18]


# Validation eval variables
# Directory to save the best model
directory = '/Users/adrianrrrrr/Documents/TFM/adrian_tfm/ASCAT_l3_collocations/2020/saved_models'
file_name = '/model'
file_ext = '.pt'
loss_val = [] # List with epoch averaged validation losses
best_val_loss = 50 # Initialisation of best loss for saving the best model
# 40% of data for validation 
X_val = in_data[18:31]
y_val = gt_data[18:31]

#TODO Implement patience for automatizing the early stopping
patience = 3 # hyperparameter to control Early stopping (To avoid over-fitting)
previous_loss = 0 # to control the loss increase
# Training - Validation loop
num_epochs = 50 # Number of total passess through training dataset
losses_index = 0 # To access the corresponding list losses
for epoch in range(num_epochs):

    # Train loop
    running_loss_train = [] # Each epoch running train losses
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

        running_loss_train.append(float(final_loss)) # List with all individual losses 

        # Gradient clipping to try to smooth the learning (It works, but experimentally seems good to let big gradients in first epochs)
        #torch.nn.utils.clip_grad_norm_(model.parameters(),1)
        # Update of the model paramters based on each gradient. It adjust parameters using Adam opt. algorithm.
        optimizer.step()


    loss_train.append(sum(running_loss_train) / len(running_loss_train))
    print(f'Epoch [{epoch+1}/{num_epochs}], training loss: {loss_train[losses_index]:.4f}')

   # Validation loop
    model.eval()
    running_loss_val = [] # Each epoch running validation losses
    with torch.no_grad():
        for image, target in zip(X_val,y_val):

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

            running_loss_val.append(float(final_loss)) # Vector of losses for debugging


    loss_val.append(sum(running_loss_val) / len(running_loss_val))

    '''
    # TODO: Automatise early stopping
    if len(loss_val) > 3:
        three_prev_loss = [loss_val[losses_index-3],loss_val[losses_index-2],loss_val[losses_index-1]]

        if loss_val[losses_index] < min(three_prev_loss):
            continue
        elif loss_val[losses_index] > min(three_prev_loss)
    '''
    if epoch == 12: # Epoch 13 with a patience = 3 gives the best loss 
        # Save the model 
        model_state_dict = model.state_dict()
        rounded_loss = "{:.6f}".format(loss_val[losses_index]) #best_loss is the best testing loss
        rounded_loss = rounded_loss[0]+'DOT'+rounded_loss[2:7]
        file_save_name = directory+file_name+'_'+str(13)+'_'+rounded_loss+file_ext
        torch.save(model_state_dict,file_save_name)
        print("Epoch 13 mode saved in: ",file_save_name)

    print(f'Epoch [{epoch+1}/{num_epochs}], validation loss: {loss_val[losses_index]:.4f}')

    losses_index=losses_index+1


# Training stats: Plotting + analysis

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,6))
x = [x for x in range(1,num_epochs+1)]
min_val = 0.5
max_val = 1.2

ax1.plot(x,loss_train,label='UNet training loss')
ax1.set_title('Train loss')
ax1.set_xlabel('epoch')
ax1.set_ylabel('loss')
#max_y = sum(loss_train)/len(loss_train)  + 0.5
#min_y = max_y - 1
ax1.set_ylim(min_val,max_val)
ax1.set_facecolor('lightgrey')
plot_text = f'Learning rate = {learning_rate}'
props = dict(boxstyle='round',facecolor='white',alpha=0.8)
ax1.text(0.6,0.9,plot_text,transform=ax1.transAxes,fontsize=9,verticalalignment='top', bbox=props)
ax1.grid(True, which='both')
ax1.minorticks_on()

ax2.plot(x,loss_val,label='UNet validation loss')
ax2.set_title('Validation loss')
ax2.set_xlabel('epoch')
ax2.set_ylabel('loss')
#max_y = sum(loss_val)/len(loss_val)  + 0.5
#min_y = max_y - 1
ax2.set_ylim(min_val,max_val)
ax2.set_facecolor('lightgrey')
plot_text = f'Learning rate = {learning_rate}'
props = dict(boxstyle='round',facecolor='white',alpha=0.8)
ax2.text(0.6,0.9,plot_text,transform=ax2.transAxes,fontsize=9,verticalalignment='top', bbox=props)
ax2.grid(True, which='both')
ax2.minorticks_on()
plt.show()

fig_dir = '/Users/adrianrrrrr/Documents/TFM/adrian_tfm/ASCAT_l3_collocations/2020'
str_lr = str(learning_rate)
save_name = fig_dir+'/train_lr_'+str_lr[0]+'DOT'+str_lr[2:]+'.png'
#plt.savefig(save_name)


'''
# TODO 
# Saving the best model on disk with the agreed naming
model_state_dict = model.state_dict()
rounded_loss = "{:.6f}".format(avg_test_loss) #best_loss is the best testing loss
rounded_loss = rounded_loss[0]+'DOT'+rounded_loss[2:7]
file_save_name = directory+file_name+'_'+str(best_epoch)+'_'+rounded_loss+file_ext
torch.save(model_state_dict,file_save_name)
'''

#wandb.finish()

