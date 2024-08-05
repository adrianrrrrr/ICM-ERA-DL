from glob import glob
import numpy as np
from netCDF4 import Dataset
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

    def read_l2_nc_data(self, fpath: str, var_names: Iterable) -> np.array:
        f = Dataset(fpath)
        selected_data = []
        hg_vars = ['se_model_wind_curl', 'se_model_wind_divergence']

        for idx, var in enumerate(var_names):
            var_data = f.variables[var].__array__()
            var_data.data[var_data.mask] = np.nan
            if var in hg_vars:
                restored_var = np.full(f.variables['lon'].__array__().shape, np.nan)
                var_data_cr = (var_data[:, 1:] + var_data[:, :-1]) / 2
                var_data_cr.mask = var_data[:, 1:].mask & var_data[:, :-1].mask
                var_data_cr = (var_data_cr[1:, :] + var_data_cr[:-1, :]) / 2
                var_data_cr.mask = var_data_cr[1:, :].mask & var_data_cr[:-1, :].mask
                var_data_cr[var_data_cr.mask] = np.nan
                restored_var[1:-1, 1:-1] = var_data_cr
                var_data = restored_var
            #print(var, var_data.shape)
            selected_data.append(var_data)
        f.close()
        data = np.ma.stack(selected_data)
        return data

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
    
    def calculate_target(self, fpath: str):
        scat_data = self.nc_reader(fpath, self.scat_model_var_names['scat'])
        model_data = self.nc_reader(fpath, self.scat_model_var_names['model'])
        targets = scat_data - model_data
        return targets
    
    def generate_dataset(self, fpath):
        #Reads netcdf swath data with shape n_variables x rows x swath_width
        input_data = self.nc_reader(fpath, self.input_var_names)
        targets = self.calculate_target(fpath)
        #Reshape the data into n_variables x number of samples
        input_data = input_data.reshape(input_data.shape[0], -1)
        targets = targets.reshape(targets.shape[0], -1)
        #Filter the data that has both u and v scat winds masked at the same time
        filter = ~np.all(targets.mask, axis=0)
        #First dimention are the variable ids, filtering by second
        input_data = input_data.data[:, filter]
        targets = targets.data[:, filter]
        if self.downsample_ratio < 1:
            if self.seed:
                np.random.seed(self.seed)
            print(f"Reducing the dataset to {self.downsample_ratio*100}%")
            total_len = targets.shape[1]
            downsampled_len = int(total_len*self.downsample_ratio)
            print(f"Randomly reducing size from {total_len} to {downsampled_len}")
            random_ids = np.random.randint(total_len, size=downsampled_len)
            input_data = input_data[:, random_ids]
            targets = targets[:, random_ids]
        input_data = input_data.transpose()
        targets = targets.transpose()
        return input_data, targets
    
    def generate_np_files(self):
        self.create_dir(self.output_dir)
        accumulated_inputs = []
        accumulated_targets = []
        out_file_counter = 0
        out_files = []
        for fpath in self.file_list:
            print(f"Processing {fpath}")
            inputs, targets = self.generate_dataset(fpath)
            # print(type(accumulated_inputs), accumulated_inputs)
            accumulated_inputs.append(inputs)
            accumulated_targets.append(targets)
            if len(np.vstack(accumulated_targets)) >= self.records_per_file:
                accumulated_inputs = np.vstack(accumulated_inputs)
                accumulated_targets = np.vstack(accumulated_targets)
                # Split array
                chunk_ids = np.arange(0, len(accumulated_inputs), self.records_per_file).astype(int)
                print(chunk_ids)
                inputs_arr = np.split(accumulated_inputs, chunk_ids)
                target_arr = np.split(accumulated_targets, chunk_ids)
                print(len(inputs_arr))
                # Save files
                for idx in range(len(inputs_arr)):
                    if len(inputs_arr[idx]) == self.records_per_file:
                        np_data_path = f"{self.output_dir}{self.separator}{self.out_file_prefix}{out_file_counter:03d}.npz"
                        print(f"saving rows inputs {inputs_arr[idx].shape} {target_arr[idx].shape} in {np_data_path}")
                        np.savez(np_data_path, input_var_names=np.array(self.input_var_names), inputs=inputs_arr[idx],
                                 targets=target_arr[idx])
                        out_file_counter += 1
                        out_files.append(np_data_path)
                    else:
                        accumulated_inputs = [inputs_arr[idx]]
                        accumulated_targets = [target_arr[idx]]

        accumulated_inputs = np.vstack(accumulated_inputs)
        accumulated_targets = np.vstack(accumulated_targets)
        np_data_path = f"{self.output_dir}{self.separator}{self.out_file_prefix}{out_file_counter:03d}.npz"
        print(f"saving rows inputs {accumulated_inputs.shape} {accumulated_targets.shape} in {np_data_path}")
        np.savez(np_data_path, input_var_names=np.array(self.input_var_names), inputs=accumulated_inputs,
                 targets=accumulated_targets)
        out_files.append(np_data_path)
        return out_files