import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dask.diagnostics import ProgressBar
from droplet_impact import utils as ut


Windfarm = 'anholt'  # Change to 'anholt' if needed

# Read the data from the file (chose the windfarm you want to use)
if Windfarm == 'billund':
    df = pd.read_csv('droplet_impact/era5_erosion_atlas/data/results/damage_increments_billund.csv')
    df_theta = pd.read_csv('droplet_impact/era5_erosion_atlas/data/results/damage_increments_billund_theta.csv')
    df_corr = pd.read_csv('droplet_impact/era5_erosion_atlas/data/results/damage_increments_billund_corr.csv')
    df_ini = pd.read_csv('droplet_impact/era5_erosion_atlas/data/results/damage_increments_billund_ini.csv')

elif Windfarm == 'anholt':
    df = pd.read_csv('droplet_impact/era5_erosion_atlas/data/results/damage_increments_anholt.csv')
    df_theta = pd.read_csv('droplet_impact/era5_erosion_atlas/data/results/damage_increments_anholt_theta.csv')
    df_corr = pd.read_csv('droplet_impact/era5_erosion_atlas/data/results/damage_increments_anholt_corr.csv')
    df_ini = pd.read_csv('droplet_impact/era5_erosion_atlas/data/results/damage_increments_anholt_ini.csv')

def plot_impinged_rain_and_damage(df_corr, df_ini, df_theta, df):
    # Sort dataframes by time
    df_corr = df_corr.sort_values('time')
    df_ini = df_ini.sort_values('time')
    df_theta = df_theta.sort_values('time')
    df = df.sort_values('time')

    # Group and sum increments
    damage_corr_sum = df_corr.groupby('time')['damage_inc'].sum().sort_index()
    damage_ini_sum = df_ini.groupby('time')['damage_inc'].sum().sort_index()
    damage_theta_sum = df_theta.groupby('time')['damage_inc'].sum().sort_index()
    damage_sum = df.groupby('time')['damage_inc'].sum().sort_index()
    # Check for negative increments
    if (damage_corr_sum < 0).any() or (damage_ini_sum < 0).any() or (damage_theta_sum < 0).any() or (damage_sum < 0).any():
        print("Warning: Negative damage increments detected.")

    # Calculate cumulative sums
    accumulated_damage_corr = damage_corr_sum.cumsum()
    accumulated_damage_ini = damage_ini_sum.cumsum()
    accumulated_damage_theta = damage_theta_sum.cumsum()
    accumulated_damage = damage_sum.cumsum()

    # Downsample
    accumulated_damage_corr_ds = accumulated_damage_corr.iloc[::100]
    accumulated_damage_ini_ds = accumulated_damage_ini.iloc[::100]
    accumulated_damage_theta_ds = accumulated_damage_theta.iloc[::100]
    accumulated_damage_ds = accumulated_damage.iloc[::100]

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(accumulated_damage_ini_ds.index, accumulated_damage_ini_ds.values, label='Initial Damage (Cumulative)', color='orange')
    plt.plot(accumulated_damage_corr_ds.index, accumulated_damage_corr_ds.values, label='Corrected Damage with only the speed reduction(Cumulative)', color='blue')
    plt.plot(accumulated_damage_theta_ds.index, accumulated_damage_theta_ds.values, label='Corrected Damage only wiht angle dependency (Cumulative)', color='green')
    plt.plot(accumulated_damage_ds.index, accumulated_damage_ds.values, label='Total Damage after correction (Cumulative)', color='red')
    plt.xlabel('Time')
    plt.ylabel('Accumulated Damage')
    plt.title('Comparison of Accumulated Damage Over Time for Billund')
    plt.legend()
    plt.grid()
    plt.show()

# Call the function to plot
plot_impinged_rain_and_damage(df_corr, df_ini, df_theta,df)