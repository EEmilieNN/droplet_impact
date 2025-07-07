import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dask.diagnostics import ProgressBar
from droplet_impact import utils as ut
import calculate_erosion_final as calc
import os

PLACE = 'anholt'  # Change to 'billund' if needed

# Read the data from the file (chose the windfarm you want to use)
if PLACE == 'billund':
    df = pd.read_csv('droplet_impact/model_complete/erosion_data/results/damage_increments_billund.csv')
    df_theta = pd.read_csv('droplet_impact/model_complete/erosion_data/results/damage_increments_billund_theta.csv')
    df_corr = pd.read_csv('droplet_impact/model_complete/erosion_data/results/damage_increments_billund_corr.csv')
    df_ini = pd.read_csv('droplet_impact/model_complete/erosion_data/results/damage_increments_billund_ini.csv')

elif PLACE == 'anholt':
    df = pd.read_csv('droplet_impact/model_complete/erosion_data/results/damage_increments_anholt.csv')
    df_theta = pd.read_csv('droplet_impact/model_complete/erosion_data/results/damage_increments_anholt_theta.csv')
    df_corr = pd.read_csv('droplet_impact/model_complete/erosion_data/results/damage_increments_anholt_corr.csv')
    df_ini = pd.read_csv('droplet_impact/model_complete/erosion_data/results/damage_increments_anholt_ini.csv')

def plot_impinged_rain_and_damage(df_corr, df_ini, df_theta, df):
    """This function plots the accumulated damage over time for the given dataframes."""
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
#plot_impinged_rain_and_damage(df_corr, df_ini, df_theta,df)

def plot_bar_chart(folder_path):
    """This function plots a bar chart of the average time before failure for each file, with and without correction."""
    
    plt.figure(figsize=(10, 6))
    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            print(f"Processing file: {filename}")
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(
                file_path,
                sep=r"\s+",
                comment="#",
                names=["year", "month", "day", "hour", "minute", "ws", "dir", "hyw", "hyT", "r10m"]
            )
            df["time"] = pd.to_datetime(df[["year", "month", "day", "hour", "minute"]])
            df = df.drop(columns=["year", "month", "day", "hour", "minute"])
            df = df.set_index("time").sort_index()

            # Create derived columns
            df["rainfall"] = df.apply(lambda row: row["r10m"] if row["hyT"] in [1, 7] else 0, axis=1)

            # Check for negative and NaN values
            neg_count = (df < 0).sum()
            nan_count = df.isnull().sum()

            # Parameters for the function
            dt = 600
            turbine_type = 'IEA_15MW'
            ws_valid = df['ws'].values
            rain_valid = df['rainfall'].values

            # Run the impinged rain function
            rainfall_b, damage_inc = calc.ds_impinged_rain_theta(ws_valid, rain_valid, dt, 'IEA_15MW')
            df['rainfall_b'] = rainfall_b
            df['damage_inc'] = damage_inc

            initial_rainfall_b, initial_damage_inc = calc.ds_impinged_rain(ws_valid, rain_valid, dt, 'IEA_15MW')
            df['initial_rainfall_b'] = initial_rainfall_b
            df['initial_damage_inc'] = initial_damage_inc

            # Mean damage
            mean_damage = df['damage_inc'].mean()
            mean_initial_damage = df['initial_damage_inc'].mean()
            t_corrected = 6*24*365/mean_damage
            t_initial = 6*24*365/mean_initial_damage

            # Plotting
            plt.bar(['Corrected Damage', 'Initial Damage'], [t_corrected, t_initial], color=['blue', 'orange'], label = os.path.splitext(filename)[0])
            plt.ylabel('Time before failure (s)')
    plt.title(f'Time to Corrected Damage vs Initial Damage for {filename}')
    plt.grid(axis='y')
    plt.show()

# Call the function to plot the bar chart
plot_bar_chart('droplet_impact/model_complete/erosion_data/meteorological_data/')