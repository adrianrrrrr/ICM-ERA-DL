import time
import numpy as np
import os
import sys
import gc

from typing import Tuple, Dict, Any, List
from torchvision import datasets, transforms

import matplotlib

import matplotlib.pyplot as plt

import mlx as ml
import mlx.core as ml # Adding support for Apple Silicon
import mlx.nn.layers as nn_mlx

from netCDF4 import Dataset as netDataset

'''
Notes by Adrian R:
We are using MPS: Metal Performance Shaders to accelerrate operations by using Apple Silicon M3 GPU 
in a nutshell refactor torch.cuda by torch.mps and so. CUDA will only be available at Cluster

Also we use mlx.core as library for hardware accelerated GPU operations. May it also Neral Unit accelerated? Check! Would be amazing
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")
    
    device = torch.device("cpu")

else:
    device = torch.device("mps")


class double_conv(nn.Module):
    '''
    Double Convolution layer with both 2 BN and Activation Layer in between
    Conv2d==>BN==>Activation==>Conv2d==>BN==>Activation
    '''
    def __init__(self, in_channel, out_channel):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1,device=device),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channel,device=device),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1,device=device),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channel,device=device)
        )
    def forward(self, x):
        x = self.conv(x)
        return x

class down_conv(nn.Module):
  '''
  A maxpool layer followed by a Double Convolution.
  MaxPool2d==>double_conv.
  '''
  def __init__(self, in_channel, out_channel):
    super(down_conv, self).__init__()
    self.down = nn.Sequential(
        nn.MaxPool2d(2),
        double_conv(in_channel, out_channel)
    )
  def forward(self, x):
    x = self.down(x)
    return x

class up_sample(nn.Module):
  def __init__(self, in_channel, out_channel):
    super(up_sample, self).__init__()
    self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    self.double_conv = double_conv(in_channel, out_channel)

  def forward(self, x1, x2):
      x1 = self.up(x1)
      x = torch.cat([x1, x2], dim=1)
      x = self.double_conv(x)
      return x

class UNet(nn.Module):
  def __init__(self, in_channel, out_channel):
    super(UNet, self).__init__()
    ## DownSampling Block (ENCODER)
    self.down_block1 = double_conv(in_channel, 16)
    self.down_block2 = down_conv(16, 32)
    self.down_block3 = down_conv(32, 64)
    self.down_block4 = down_conv(64, 128)
    self.down_block5 = down_conv(128, 256)
    self.down_block6 = down_conv(256, 512)
    #self.down_block7 = down_conv(512, 1024)

    ## UpSampling Block (DECODER)
    #self.up_block1 = up_sample(1024+512, 512)
    self.up_block2 = up_sample(512+256, 256)
    self.up_block3 = up_sample(256+128, 128)
    self.up_block4 = up_sample(128+64, 64)
    self.up_block5 = up_sample(64+32, 32)
    self.up_block6 = up_sample(32+16, 16)
    self.up_block7 = nn.Conv2d(16, out_channel, kernel_size=1,device=device)


  def forward(self, x):
    #Down
    x1 = self.down_block1(x)
    x2 = self.down_block2(x1)
    x3 = self.down_block3(x2)
    x4 = self.down_block4(x3)
    x5 = self.down_block5(x4)
    x6 = self.down_block6(x5)
    #x7 = self.down_block7(x6)
    #Up

    #x8 = self.up_block1(x7, x6)
    x7 = self.up_block2(x6, x5)
    x8 = self.up_block3(x7, x4)
    x9 = self.up_block4(x8, x3)
    x10 = self.up_block5(x9, x2)
    x11 = self.up_block6(x10, x1)
    x12 = self.up_block7(x11)
    # out = torch.sigmoid(x12) # CLASSIFICATON
    out = x12 # REGRESSION
    return out
  

# Dataload helper function + Data normalization as norm_data = (data-mean)/std
# TODO: parameters: file_dir, number_files,... or authomatise it
def MyDataLoader():
  start_time = time.time()

  input_var_names = ['lon', 'lat',
                    'eastward_model_wind', 'northward_model_wind', 'model_speed', 'model_dir',
                    'msl', 'air_temperature', 'q', 'sst', 'uo', 'vo']
  output_var_names = ['u','v']

  train_input_folder =  "/Volumes/SSD Adrian/TFM/adrian_tfm/ASCAT_l3_collocations/2020/train/"
  loader_input_var_names = ['eastward_model_wind', 'northward_model_wind', 'model_speed', 'model_dir', 
                              'msl', 'air_temperature', 'q', 'sst', 'uo', 'vo']

  input_data = []
  ground_truth = []

  for day in range(1,10):
      all_data = []
      train_input_file = train_input_folder+"ascata_2020010"+str(day)+"_l3_asc.nc"
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
      X = input_masked_data[:,864:1120,2568:2824]

      input_data.append(X)
      
      # Ground truth (y)
      u = f.variables['eastward_wind'].__array__()[0]
      v = f.variables['northward_wind'].__array__()[0]
      u_model = f.variables['eastward_model_wind'].__array__()[0]
      v_model = f.variables['northward_model_wind'].__array__()[0]
      f.close()
      targets = np.ma.MaskedArray([u - u_model, v - v_model])
      #targets = np.transpose(targets,(1,2,0))
      y = targets[:,864:1120,2568:2824]
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
# Adrian's Note: np.ma.mean works flawlessly! Test it in EDA, maybe the differences in mean
# are beause of the used method (This one seems to works perfectly)
def MyNorm(BatchedData,NumberExamples):
   for example in range(0,NumberExamples):
    for variable in range(0,len(BatchedData[example])):
        mean = np.ma.mean(BatchedData[example][variable])
        std = np.ma.std(BatchedData[example][variable])
        BatchedData[example][variable] = (BatchedData[example][variable]-mean)/std


   
