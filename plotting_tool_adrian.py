from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

def make_colorbar(ax, mappable, **kwargs):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib as mpl
    divider = make_axes_locatable(ax)
    orientation = kwargs.pop('orientation', 'vertical')
    if orientation == 'vertical':
        loc = 'right'
    elif orientation == 'horizontal':
        loc = 'bottom'
    cax = divider.append_axes(loc, '3%', pad='3%', axes_class=mpl.pyplot.Axes)
    ax.get_figure().colorbar(mappable, cax=cax, orientation=orientation)

fn = "C:\\tmp\\ascata_20200101_l3_asc.nc"

f = Dataset(fn)
lon = f.variables['lon'].__array__()
lat = f.variables['lat'].__array__()
u = f.variables['eastward_wind'].__array__()
v = f.variables['northward_wind'].__array__()
u_model = f.variables['eastward_model_wind'].__array__()
v_model = f.variables['northward_model_wind'].__array__()
f.close()

u_bias = u - u_model
v_bias = v - v_model

fig = plt.figure(figsize=(15, 10))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0.5, linestyle='--')
im = ax.pcolor(lon[2568:2824], lat[864:1120], u[0, 864:1120, 2568:2824], cmap='bwr', vmin=-10, vmax=10)
plt.title(f"ASCAT-A u-component asc 01/01/2020 ", fontsize=20)
make_colorbar(ax, im, orientation='vertical')
plt.savefig(f'ASCAT_u_component_area_TFM.png', bbox_inches='tight', dpi=600)
plt.show()

fig = plt.figure(figsize=(15, 10))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0.5, linestyle='--')
im = ax.pcolor(lon[2568:2824], lat[864:1120], v[0, 864:1120, 2568:2824], cmap='bwr', vmin=-10, vmax=10)
plt.title(f"ASCAT-A v-component asc 01/01/2020 ", fontsize=20)
make_colorbar(ax, im, orientation='vertical')
plt.savefig(f'ASCAT_v_component_area_TFM.png', bbox_inches='tight', dpi=600)
plt.show()

fig = plt.figure(figsize=(15, 10))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0.5, linestyle='--')
im = ax.pcolor(lon[2568:2824], lat[864:1120], u_bias[0, 864:1120, 2568:2824], cmap='bwr', vmin=-2, vmax=2)
plt.title(f"u-bias ASCAT-A - ERA5 asc 01/01/2020 ", fontsize=20)
make_colorbar(ax, im, orientation='vertical')
plt.savefig(f'u_bias_area_TFM.png', bbox_inches='tight', dpi=600)
plt.show()


fig = plt.figure(figsize=(15, 10))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0.5, linestyle='--')
im = ax.pcolor(lon[2568:2824], lat[864:1120], v_bias[0, 864:1120, 2568:2824], cmap='bwr', vmin=-2, vmax=2)
plt.title(f"v-bias ASCAT-A - ERA5 asc 01/01/2020 ", fontsize=20)
make_colorbar(ax, im, orientation='vertical')
plt.savefig(f'v_bias_area_TFM.png', bbox_inches='tight', dpi=600)
plt.show()