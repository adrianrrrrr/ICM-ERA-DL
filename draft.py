import torch
import torch.nn as nn
import torch.nn.functional as F

if torch.cuda.is_available():
    print("Cuda is available. There are ",torch.cuda.device_count()," devices")
    print("Current device is ",torch.cuda.current_device()," named: ",torch.cuda.get_device_name(0))
# Cuda is available. There are  1  devices
# Current device is  0  named:  NVIDIA GeForce RTX 2080 Ti

else:
    print("Fail")


