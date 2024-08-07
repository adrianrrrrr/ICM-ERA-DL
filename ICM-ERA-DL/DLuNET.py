from glob import glob
import numpy as np
from collections.abc import Iterable
import os
 
class uNETInputGenerator:
    def __init__(self, input_var_names, target_var_names, input_grid='l3', input_dir=None,
                 output_dir=None, file_list=None, downsample_ratio=1, records_per_file=1e9, out_file_prefix="", seed=None, n_sigma=None):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.scat_model_var_names = {'scat': ['eastward_wind', 'northward_wind'],
                                    'model': ['eastward_model_wind', 'northward_model_wind']}
        self.input_var_names = input_var_names
        self.target_var_names = target_var_names
        self.separator = '' # Noise. It was \\ (Incorrect). / is correct for unix, but is redundant
        self.downsample_ratio = downsample_ratio
        self.n_sigma = n_sigma
        self.seed = seed
        self.records_per_file = records_per_file
        self.out_file_prefix = out_file_prefix
        if input_dir:
            self.file_list = glob(input_dir + '*.nc')
            self.file_list.sort()
        else:
            self.file_list = file_list
        if input_grid == "l3":
            self.nc_reader = self.read_l3_nc_data
        else:
            self.nc_reader = self.read_l2_nc_data
    def create_dir(self, dir): # sudo chmod a+rw /path/to/folder/that/you/want/to/write/to 
        if not os.path.exists(dir):
            os.makedirs(dir)

    def read_l3_nc_data(self, fpath: str, var_names: Iterable) -> np.array:
        f = Dataset(fpath)
        selected_data = []
        lon = f.variables['lon'].__array__()
        #lon = lon[2568:2824] # Eugenia 31/07 mail
        lat = f.variables['lat'].__array__()
        #lat = lat[864:1120] # Eugenia 31/07 mail
        lats_mesh, lons_mesh = np.meshgrid(lat, lon, indexing='ij')
        for idx, var in enumerate(var_names):
            print(idx, var)
            if var == 'lon':
                #print("adding lon mesh")
                selected_data.append(lons_mesh)
            elif var == 'lat':
                #print("adding lat mesh")
                selected_data.append(lats_mesh)
            else:
                var_data = f.variables[var].__array__()
                mask = var_data.mask
                var_data.data[mask] = np.nan
                selected_data.append(np.squeeze(var_data, axis=0))
        f.close()
        data = np.ma.stack(selected_data)
        return data
    
  