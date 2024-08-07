from DLuNET import *
from netCDF4 import Dataset
import time
#import numpy as np
import matplotlib.pyplot as plt

start = time.time()

all_data = []
input_var_names = [#'lon', 'lat', will be loaded separately
                   'eastward_model_wind', 'northward_model_wind', 'model_speed', 'model_dir',
                   'msl', 'air_temperature', 'q', 'sst', 'uo', 'vo']

train_input_file = "/Volumes/SSD iMac/TFM/adrian_tfm/ASCAT_l3_collocations/2020/train/ascata_20200101_l3_asc.nc"

# TODO: Automatic load just creating the pointer to the directory
f = Dataset(train_input_file)

lon = f.variables['lon'].__array__() #0-360 degrees = 2880 points
lat = f.variables['lat'].__array__() #0-180 degrees = 1440 points

lons, lats = np.meshgrid(lon, lat)
all_data.append(lons)
all_data.append(lats)

for var_name in input_var_names:
    var_data = f.variables[var_name].__array__()[0]
    all_data.append(var_data)

input_masked_data = np.stack(all_data)

u = f.variables['eastward_wind'].__array__()[0]
v = f.variables['northward_wind'].__array__()[0]
u_model = f.variables['eastward_model_wind'].__array__()[0]
v_model = f.variables['northward_model_wind'].__array__()[0]
f.close()

targets = np.stack([u - u_model, v - v_model])

# Let's define some hyper-parameters
hparams = {
    'batch_size': 64,
    'num_epochs': 10,
    'test_batch_size': 64,
    'learning_rate': 1e-3,
    'log_interval': 100,
}

# This is the unmasked train data (Input). In Shape = (995807, 12)
train = train_data['inputs']
# These are the input var names in a numpy.ndarray. Equivalent to the list input_var_names. Shape = (12,)
ix = train_data['input_var_names']
# This is the unmasked ground truth data (Input). Out Shape = (995807, 2)
ground_truth = targets # Just to align with previous code. Delete it in the future


for index in range(12):
    mean = np.nanmean(train[:,index]) #Computes ignoring NaN values
    std = np.nanstd(train[:,index]) #Computes ignoring NaN values
    maxd = train[:,index].max()
    mind = train[:,index].min()
    print(ix[index]," variable mean =  ",mean, " ; std = ",std," ; max = ",maxd," min = ",mind)

'''
plt.hist(train[:,9], bins='auto')  # arguments are passed to np.histogram
plt.title("Histogram of SST with 'auto' bins")
plt.show()

aux = ['u','v']
for index in range(2):
    mean = np.nanmean(ground_truth[:,index]) #Computes ignoring NaN values
    std = np.nanstd(ground_truth[:,index]) #Computes ignoring NaN values
    maxd = ground_truth[:,index].max()
    mind = ground_truth[:,index].min()
    print(aux[index]," ground truth mean =  ",mean, " ; std = ",std," ; max = ",maxd," min = ",mind)

plt.hist(ground_truth[:,0], bins='auto')  # arguments are passed to np.histogram
plt.title("Histogram of u component ground truth with 'auto' bins")
plt.show()

plt.hist(ground_truth[:,1], bins='auto')  # arguments are passed to np.histogram
plt.title("Histogram of v component ground truth with 'auto' bins")
plt.show()

plt.imshow(train_data['inputs'])
plt.colorbar()
plt.show()
'''