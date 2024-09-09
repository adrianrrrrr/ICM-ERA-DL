import time
import numpy as np
import os
import sys
import gc

from typing import Tuple, Dict, Any, List


import matplotlib
import matplotlib.pyplot as plt

from netCDF4 import Dataset as netDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset, random_split

# Check graphic card acceleration
if torch.cuda.is_available():
    mydevice = torch.device("cuda")
    torch.cuda.empty_cache()
    print("Cuda is available. There are ",torch.cuda.device_count()," devices")
    print("Current device is ",torch.cuda.current_device()," named: ",torch.cuda.get_device_name(0))
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    mydevice = torch.device("mps")
    print("MPS is available")
else:
    mydevice = torch.device("cpu")

# UNET declaration for input of 256x256x1 and output of 256x256x2
# 
# Generated by Copilot, verified by Medium article (Its the same) + Adrian inspection step by step
# checked it works for classification
# TODO: Generalise number of channels and so
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        # Contracting path
        self.enc1 = self.conv_block(12, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.enc5 = self.conv_block(512, 1024) # Bottleneck / latent space
        
        # Expansive path
        self.upconv4 = self.upconv_block(1024, 512)
        self.dec4 = self.conv_block(1024, 512)
        self.upconv3 = self.upconv_block(512, 256)
        self.dec3 = self.conv_block(512, 256)
        self.upconv2 = self.upconv_block(256, 128)
        self.dec2 = self.conv_block(256, 128)
        self.upconv1 = self.upconv_block(128, 64)
        self.dec1 = self.conv_block(128, 64)
        
        # Output layer. Linear activation makes it Regression, not Classification
        self.out_conv = nn.Conv2d(64, 2, kernel_size=1)
    
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1,device=mydevice),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1,device=mydevice),
            nn.ReLU(inplace=True)
        )
    
    def upconv_block(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    
    def forward(self, x):
        # Contracting path
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))
        enc5 = self.enc5(F.max_pool2d(enc4, 2))
        
        # Expansive path
        dec4 = self.upconv4(enc5)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)
        
        out = self.out_conv(dec1)
        return out
    
# Dataload helper function + Data normalization as norm_data = (data-mean)/std
# TODO: parameters: file_dir, number_files,... or authomatise it
'''
Currently, this function loads 9 days (from ascata_20200101_l3_asc.nc to _20200109_)
'''
def MyDataLoader():
  start_time = time.time()

  train_input_folder =  "/mnt/work/datasets/ECMWF/ASCAT_l3_collocations/2020"
  loader_input_var_names = ['eastward_model_wind', 'northward_model_wind', 'model_speed', 'model_dir', 
                              'msl', 'air_temperature', 'q', 'sst', 'uo', 'vo']

  input_data = []
  ground_truth = []

  for day in range(1,10):
      all_data = []
      train_input_file = train_input_folder+"/ascata_2020010"+str(day)+"_l3_asc.nc"
      f = netDataset(train_input_file)
      
      # Creating 2D variables from 1D data 
      lon = f.variables['lon'].__array__() #0-360 degrees = 2880 points
      lat = f.variables['lat'].__array__() #0-180 degrees = 1440 points
      lons, lats = np.meshgrid(lon, lat)
      all_data.append(lons)
      all_data.append(lats)

      for var_name in loader_input_var_names:
          var_data = f.variables[var_name].__array__()[0]
          all_data.append(var_data)

      # Input data load (X) 
      input_masked_data = np.ma.MaskedArray(all_data)
      #input_masked_data = np.transpose(input_masked_data,(1,2,0)) # Putting the "channels" in the last dimension
      
      # 256x256 piece of image we agreed
      #X = input_masked_data[:,864:1120,2568:2824]
      X = input_masked_data

      input_data.append(X)
      
      # Ground truth (y)
      u = f.variables['eastward_wind'].__array__()[0]
      v = f.variables['northward_wind'].__array__()[0]
      u_model = f.variables['eastward_model_wind'].__array__()[0]
      v_model = f.variables['northward_model_wind'].__array__()[0]
      f.close()
      targets = np.ma.MaskedArray([u - u_model, v - v_model])
      #targets = np.transpose(targets,(1,2,0))
      #y = targets[:,864:1120,2568:2824]
      y = targets
      
      ground_truth.append(y)

      print(train_input_file," loaded succesfully")

  # let's free some RAM
  del all_data, lon, lat, var_name, input_masked_data, X, u, v, u_model, v_model, targets, y

  gc.collect() # Forcing garbage collection i.e. free RAM references from del listed variables

  end_time = time.time()
  elapsed_time = end_time - start_time

  memory_size = sys.getsizeof(input_data) + sys.getsizeof(ground_truth) # Bytes
  memory_size = memory_size 

  print("Dataload took ",elapsed_time," seconds")
  print("Dataset has ",memory_size,"B allocated in RAM")

  return input_data, ground_truth

# Function to normalise data
# TODO: Circular normalization, combined means / variance by mathematical formula,...
# Global vars
input_var_names = ['lon', 'lat',
                'eastward_model_wind', 'northward_model_wind', 'model_speed', 'model_dir',
                'msl', 'air_temperature', 'q', 'sst', 'uo', 'vo']
output_var_names = ['u','v']

var_stats = {var:{'mean':None,'std':None} for var in input_var_names}
gt_stats = {var:{'mean':None,'std':None} for var in output_var_names}

def MyNorm(BatchedData):
    debug = 0

   # Dict initialisation
    for index_batch, example in enumerate(BatchedData): # Just iterates over all the batched examples
        for index_var, variable_img in enumerate(example): # Enumerate gives the index from the 'example' iterable element

            if len(example) > 2 : # More than 2 variables = input data. 2 variables = ground truth data
                print("Input var normalization: ")
                if var_stats[input_var_names[index_var]]['mean'] == None and var_stats[input_var_names[index_var]]['std'] == None:
                    var_stats[input_var_names[index_var]]['mean'] = np.ma.mean(variable_img)
                    var_stats[input_var_names[index_var]]['std'] = np.ma.std(variable_img)
                    print(debug)
                    debug+=1
                # else: Update with the mathematical formula of joint mean/variance of two distributions. Discuss in meeting first. From now I leave it as this

                print("Batch index assigned = ",index_batch,". Var index assigned = ",index_var)
                BatchedData[index_batch][index_var] = (BatchedData[index_batch][index_var]-
                                                    var_stats[input_var_names[index_var]]['mean'])/var_stats[input_var_names[index_var]]['std']
            else:
                print("Ground truth normalization: ")
                if gt_stats[output_var_names[index_var]]['mean'] == None and gt_stats[output_var_names[index_var]]['std'] == None:
                    gt_stats[output_var_names[index_var]]['mean'] = np.ma.mean(variable_img)
                    gt_stats[output_var_names[index_var]]['std'] = np.ma.std(variable_img)
                    print(debug)
                    debug+=1
                # else: Update with the mathematical formula of joint mean/variance of two distributions. Discuss in meeting first. From now I leave it as this

                print("Batch index assigned = ",index_batch,". Var index assigned = ",index_var)
                BatchedData[index_batch][index_var] = (BatchedData[index_batch][index_var]-
                                                    gt_stats[output_var_names[index_var]]['mean'])/gt_stats[output_var_names[index_var]]['std']

