#from DLuNET import *
from netCDF4 import Dataset
import time
import numpy as np
import matplotlib.pyplot as plt
import torch

start = time.time()

all_data = []
input_var_names = ['lon', 'lat',
                   'eastward_model_wind', 'northward_model_wind', 'model_speed', 'model_dir',
                   'msl', 'air_temperature', 'q', 'sst', 'uo', 'vo']
output_var_names = ['u','v']

train_input_file = "/Volumes/SSD iMac/TFM/adrian_tfm/ASCAT_l3_collocations/2020/train/ascata_20200101_l3_asc.nc"

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

u = f.variables['eastward_wind'].__array__()[0]
v = f.variables['northward_wind'].__array__()[0]
u_model = f.variables['eastward_model_wind'].__array__()[0]
v_model = f.variables['northward_model_wind'].__array__()[0]
f.close()

# Ground truth: y 
targets = np.ma.MaskedArray([u - u_model, v - v_model])

for variable_index in range(12):
    var_name = input_var_names[variable_index]
    mean = np.nanmean(input_masked_data[variable_index]) #Computes ignoring NaN values
    std = np.nanstd(input_masked_data[variable_index]) 
    maxd = np.nanmax(input_masked_data[variable_index]) 
    mind = np.nanmin(input_masked_data[variable_index]) 
    print(var_name," variable mean =  ",mean, " ; std = ",std," ; max = ",maxd," min = ",mind)

    flattened_data = input_masked_data[variable_index].compressed() # This remove the masked values
    plt.hist(flattened_data, bins='auto')  # arguments are passed to np.histogram
    #plt.vlines(0,0,30000,colors="red")
    plt.title("Histogram of "+var_name+" with 'auto' bins")
    plt.show()

for component in range(2):
    comp_name = output_var_names[component]
    mean = np.nanmean(targets[component]) 
    std = np.nanstd(targets[component]) 
    maxd = np.nanmax(targets[component]) 
    mind = np.nanmin(targets[component]) 
    print(comp_name," ground truth mean =  ",mean, " ; std = ",std," ; max = ",maxd," min = ",mind)

    flattened_data = targets[component].compressed() # This remove the masked values
    plt.hist(flattened_data, bins='auto')  # arguments are passed to np.histogram
    plt.title("Histogram of "+comp_name+" with 'auto' bins")
    plt.show()