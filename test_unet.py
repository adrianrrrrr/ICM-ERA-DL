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

# Loading the test data (First days of January 2019)
num_days = 9
dir_path = "/Users/adrianrrrrr/Documents/TFM/adrian_tfm/ASCAT_l3_collocations/2019/test"
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

# Loading the best trained model TODO: input() for the name of the best model
model_path = '/Users/adrianrrrrr/Documents/TFM/adrian_tfm/ASCAT_l3_collocations/2020/saved_models'
model_name = '/model_13_0DOT81363.pt' # Best model name model_<BESTEPOCH>_<BESTLOSS>.pt
model = UNet() # Model initialization
model = model.to(mydevice) # To GPU if available
model.load_state_dict(torch.load(model_path+model_name)) # Load the trained params from train_unet.py best model 
criterion = nn.MSELoss(reduction='none') # Loss function for regression -> MSE. No reduction (Neccessary for masking values)


# Test eval variables
imag_process = 1
best_testing_loss = 100000 # Initialisation of best loss for saving the best model
running_loss_test = []
model.eval() # Setting the model to eval mode 
# Testing loop
with torch.no_grad():
    # From now batch size is 1 (One image injected to UNET, then loss is computed)
    for image, target in zip(in_data,gt_data):
        print("Processing image nr, ",imag_process)
        imag_process=imag_process+1
        input = image[None,:,864:1120,2568:2824].to(mydevice) # 256x256 patch + adding first dummy dimension for the UNET
        output = model(input) 
        groundt = target[None,:,864:1120,2568:2824].to(mydevice)

        # Create the mask for ignoring the zero values in the targets
        mask = groundt != 0
        loss = criterion(output,groundt)
        masked_loss = loss * mask # Apply the mask
        final_loss = masked_loss.sum() / mask.sum() # Normalize by the number of non-zero elements

        running_loss_test.append(float(final_loss)) # List with all individual losses 

        print(f'Loss for image #:{imag_process} = {final_loss:.4f}')
   
testing_loss = sum(running_loss_test) / len(running_loss_test)
print("Testing complete")
print("Testing loss: ","{:.6f}".format(testing_loss))

learning_rate = 0.001
# Testing stats: Plotting + analysis
#fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,6))
x = [x for x in range(1,num_days+1)]
min_val = 0
max_val = 0.9

plt.plot(x,running_loss_test,label='UNet testing error')
plt.title('Testing running loss')
plt.xlabel('image')
plt.ylabel('error')
#max_y = sum(loss_train)/len(loss_train)  + 0.5
#min_y = max_y - 1
plt.ylim(min_val,max_val)
plt.gca().set_facecolor('lightgrey')
plot_text = f'Testing loss = {testing_loss:.3f}'
props = dict(boxstyle='round',facecolor='white',alpha=0.8)
plt.text(0.6,0.9,plot_text,transform=plt.gca().transAxes,fontsize=9,verticalalignment='top', bbox=props)
plt.grid(True, which='both')
plt.minorticks_on()

plt.show()

fig_dir = '/Users/adrianrrrrr/Documents/TFM/adrian_tfm/ASCAT_l3_collocations/2019'
str_lr = str(learning_rate)
save_name = fig_dir+'/test_lr_'+str_lr[0]+'DOT'+str_lr[2:]+'.png'
plt.savefig(save_name)

# Let's get the best prediction (Day 2) to analyse the image
model.eval()
with torch.no_grad():
    input = in_data[1] # Second day image
    input = input[None,:,864:1120,2568:2824].to(mydevice) # 256x256 patch + adding first dummy dimension for the UNET
    output = model(input)
    groundt = gt_data[1] # Second day ground truth (ASCATA - ERA differences)
    groundt = groundt[None,:,864:1120,2568:2824].to(mydevice)

    # Create the mask for ignoring the zero values in the targets
    mask = groundt != 0
    loss = criterion(output,groundt)
    masked_loss = loss * mask # Apply the mask
    final_loss = masked_loss.sum() / mask.sum() # Normalize by the number of non-zero elements





