# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dask.diagnostics import ProgressBar

# %% Define input parameters
dt = 3600   # The time step of the input data in seconds

# %% Functions

def droplet_diameter(rain_rate, percentile=50):
    """Best formula for droplet diameter for 50th percentile of the droplet distribution.
    
    Args:
        rain_rate (array, float): rain rate in [mm/h]
        percentile (float, int): percentile of the droplet size in range 0-100%
    
    Returns: 
        d50: droplet diameter in [mm]
    """
    a_par, p, n = 1.3, 0.232, 2.25
    a = a_par*rain_rate**p
    d50 = (-np.log(percentile/100))**(1/n)*a
    return d50 

def terminal_velocity(droplet_diameter):
    """A fitted function to the Best data with droplet terminal velocity as function of droplet diameter

    Args:
        droplet_diameter (array_type): Droplet diameter in mm

    Returns:
        v_term (array_type): The terminal velocity of the droplets in m/s
    """
    dd = droplet_diameter
    v_term = 0.0481 * dd**3 - 0.8037 * dd**2 + 4.621 * dd
    return v_term

def tip_speed(wsp, turbine):
    """A function that calculates tip speed from wind speed.

    Args:
        wsp (array_like): Hub height wind speed input
        turbine (string): 'NREL_5MW', 'DTU_10MW', or 'IEA_15MW'

    Returns:
        ts_inter (array_like): The tip speed of the selected wind turbine
    """
    ts_file = 'tip_speed/' + turbine + '.csv'
    ts_data = pd.read_csv(ts_file)
    ts_inter = np.interp(wsp, ts_data['wsp'], ts_data['tip_speed'], left=0, right=0)
    return ts_inter
    
def impinged_rain(wsp, rainfall, dt, turbine, c=2.85e22, m = 10.5):
    """A function that calculates impinged rain on a blade tip and corresponding damage increments

    Args:
        wsp (arra_type): Wind speed at hub height [m/s]
        rainfall (array_type): The rainfall in [m]
        dt (int or float): The time step of the wsp and rainfall data [s]
        turbine (string):'NREL_5MW', 'DTU_10MW', or 'IEA_15MW'
        c (float, optional): Rain erosion test power-law constant. Defaults to 2.85e22.
        m (float, optional): Rain erosion test power-law constant. Defaults to 10.5.

    Returns:
        rainfall_b (array_like): The impinged rainfall on the blade tip [m] 
        damage_inc (array_like): Tha damage increments. When they sum up to one damage begins to occur
    """
    v_tip = tip_speed(wsp, turbine)
    # Rain rate in SI units
    rain_rate_si = rainfall/dt
    # Rain rate in [mm/h]
    rain_rate = rain_rate_si*1000*3600
    droplet_d = droplet_diameter(rain_rate)
    v_term = terminal_velocity(droplet_d)

    # Water content in a unit volume of air [-]
    water_content = xr.where(rain_rate_si > 0, rain_rate_si/v_term, 0)
    # Rain rate on the blade tip [m/s] (SI unit)
    rain_rate_b = water_content * v_tip # [m/s]
    # Impinged rainfall on the blade tip [m]
    rainfall_b = rain_rate_b*dt

    # Average impact velocity of the rain drops relative to the blade (averaging out vertical velocity)
    v_rel = np.sqrt(wsp**2 + v_tip**2)
    # Modelled impingement to reach damage in RET 
    # Impingement to damge as function of the relative velocity
    impingement = xr.where(v_rel>0, c*v_rel**(-m), np.inf)
    # Damage increments
    damage_inc = rainfall_b/impingement
    return rainfall_b, damage_inc

# %% Read in the ERA5 data
file_path = '/groups/reanalyses/era5/app/'
lat_slice = slice(53,72)
lon_slice = slice(0,32)
time_slice = slice('2010-01-01T00:00:00.000000000','2019-12-31T23:00:00.000000000')
hub_height = 150

ds_pre = xr.open_zarr(file_path + 'era5_precip.zarr')
ds_ws = xr.open_zarr(file_path + 'era5.zarr')
# %% Make a new dataset for the output
ds = xr.Dataset()
ds['ws'] = ds_ws.WS.sel(height=hub_height, 
                        latitude=lat_slice, 
                        longitude=lon_slice, 
                        time=time_slice)
ds['tp'] = ds_pre.tp.sel(latitude=lat_slice, 
                         longitude=lon_slice, 
                         time=time_slice)
ds['ptype'] = ds_pre.ptype.sel(latitude=lat_slice, 
                         longitude=lon_slice, 
                         time=time_slice)
# ptype=1: rain, ptype=7: mixture of rain and snow
ds['rainfall'] = xr.where((ds.ptype == 1) | (ds.ptype == 7), ds.tp, 0)

# %% Check for negative and nan values for rain and wind speed
neg_count = (ds < 0).sum(dim=['time', 'latitude', 'longitude'])
nan_count = ds.isnull().sum(dim=['time', 'latitude', 'longitude'])
# ds = ds.where(ds >0, 0)

# %% Calculate the damgae increments with apply_ufunc
"""If the selected area is very big this step needs to be done in loops
with smaller chunks of data"""
test0, test1 = xr.apply_ufunc(impinged_rain, # the function
                              ds['ws'], ds['rainfall'], dt, 'IEA_15MW', #args
                              input_core_dims=[['time'], ['time'], [],[]],
                              output_core_dims=[['time'], ['time']],
                              vectorize=True,
                              dask='allowed')

# %%
ds['impinged_rain'] = test0.transpose('time', 'latitude', 'longitude')
ds['damage_inc'] = test1.transpose('time', 'latitude', 'longitude')

# %%
# delayed_obj = ds.to_netcdf('damage_increments.nc', compute=False)
# with ProgressBar():
#     results = delayed_obj.compute()

# %%
date_diff = ds.time[-1].values - ds.time[0].values
nr_years = date_diff / np.timedelta64(1, 'D')/365
nr_years = round(nr_years, 1)

# %% Calculate atlas layers 
al = xr.Dataset()
al['annual_rainfall'] = ds.rainfall.sum(dim='time')/nr_years
al['annual_impinged_rainfall'] = ds.impinged_rain.sum(dim='time')/nr_years
al['erosion_time'] = nr_years/ds.damage_inc.sum(dim='time')
al['wsp_150m'] = ds.ws.mean(dim='time')
al['annual_increment_sum'] = ds.damage_inc.sum(dim='time')/nr_years

# %%
delayed_obj = al.to_netcdf('atlas_layers.nc', compute=False)
with ProgressBar():
    results = delayed_obj.compute()

# %%
