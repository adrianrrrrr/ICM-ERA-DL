from netCDF4 import Dataset as netDataset
import time
import numpy as np
import sys
import gc

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