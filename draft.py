import torch
import torch.nn as nn
import torch.nn.functional as F

'''
if torch.cuda.is_available():
    print("Cuda is available. There are ",torch.cuda.device_count()," devices")
    print("Current device is ",torch.cuda.current_device()," named: ",torch.cuda.get_device_name(0))
# Cuda is available. There are  1  devices
# Current device is  0  named:  NVIDIA GeForce RTX 2080 Ti

else:
    print("Fail")
'''

file_path = '/home/usuaris/imatge/adrian.ramos/saving_file.txt'
# Open a file in write mode
with open(file_path, 'w') as file:
    # Write some text to the file
    file.write('Hello, world!\n')
    file.write('No prob in writting files with Slurm! (Disks virtually mounted).\n')

print('File written successfully.')
