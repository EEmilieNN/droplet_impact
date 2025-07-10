# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from droplet_impact import utils as ut
from dask.diagnostics import ProgressBar

# %% Define input parameters
dt = 600   # The time step of the input data in seconds

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
    ts_file = 'droplet_impact/model_complete/erosion_data/tip_speed/' + turbine + '.csv'
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

def ds_impinged_rain(wsp, rainfall, dt, turbine, droplet_d=None, v_term=None, drop_dep='softsign'):
    """A function that calculates impinged rain on a blade tip and corresponding damage increments
 
    Args:
        wsp (arra_type): Wind speed at hub height
        rainfall (array_type): The rainfall in [m]
        dt (int or float): The time step of the wsp and rainfall data [s]
        turbine (string):'NREL_5MW', 'DTU_10MW', or 'IEA_15MW'
 
    Returns:
        rainfall_b (array_like): The impinged rainfall on the blade tip [m]
        damage_inc (array_like): Tha damage increments. When they sum up to one damage begins to occur
    """
    # Wind profile power law (Wind speed at 150 m)
    wsp150 = wsp * (150/10)**(0.143)
    # Calculate the tip speed (the 0.998 factor is the fraction that defines where we are on the blade)
    v_tip = tip_speed(wsp150, turbine)*0.998
    # Rain rate in SI units [m/s]
    rain_rate_si = rainfall/(1000*dt)
    # Rain rate in [mm/h]
    rain_rate = rain_rate_si*1000*3600
    if droplet_d is None:
        droplet_d = droplet_diameter(rain_rate)
    if v_term is None:
        v_term = terminal_velocity(droplet_d)
    # Water content in a unit volume of air [-]
    water_content = np.zeros(rain_rate.shape)
    # If the terminal velocity is zero, we cannot divide by it
    valid = np.where(v_term > 0)[0]
    water_content[valid] = np.divide(rain_rate_si[valid], v_term[valid])
    # Rain rate on the blade tip [m/s] (SI unit)
    rain_rate_b = water_content * v_tip # [m/s]
    # Impinged rainfall on the blade tip [m]
    rainfall_b = rain_rate_b*dt
    # Average impact velocity of the rain drops relative to the blade (maximum impact speed)
    v_rel = np.sqrt(wsp150**2 + (v_tip)**2)
    # The droplet dependant c and m parameters
    if drop_dep == 'softsign':
        m_imp = -3.134*(droplet_d-2.107) / (1 + np.abs(droplet_d-2.107)) + 8.870
        imp_100 = -17.066*(droplet_d-2.294) / (1 + np.abs(droplet_d-2.294)) + 21.738
    elif drop_dep == 'linear':
        m_imp = -1.284 * droplet_d + 11.510
        imp_100 = -6.604 * droplet_d + 36.955
    c_imp = imp_100*100**m_imp
    # Impingement to damage as function of the relative velocity
    impingement = np.where(v_rel>0, c_imp*v_rel**(-m_imp), np.inf)
    # Damage increments
    damage_inc = np.divide(rainfall_b,impingement)
    return rainfall_b, damage_inc

def ds_impinged_rain_theta(wsp, rainfall, dt, turbine, droplet_d=None, v_term=None, drop_dep='softsign'):
    """
    FINAL FUNCTION
    A function that calculates impinged rain on a blade tip and corresponding damage increments with angle dependency and speed correction.
 
    Args:
        wsp (arra_type): Wind speed at hub height
        rainfall (array_type): The rainfall in [m]
        dt (int or float): The time step of the wsp and rainfall data [s]
        turbine (string):'NREL_5MW', 'DTU_10MW', or 'IEA_15MW'
 
    Returns:
        rainfall_b (array_like): The impinged rainfall on the blade tip [m]
        damage_inc (array_like): Tha damage increments. When they sum up to one damage begins to occur
    """
    # Wind profile power law
    wsp150 = wsp * (150/10)**(0.143)
    # Rain rate in SI units [m/s]
    rain_rate_si = rainfall/(1000*dt)
    # Rain rate in [mm/h]
    rain_rate = rain_rate_si*1000*3600
    if droplet_d is None:
        droplet_d = droplet_diameter(rain_rate)
    if v_term is None:
        v_term = terminal_velocity(droplet_d)
    # Calculate the tip speed (the 0.998 factor is the fraction that defines where we are on the blade)
    v_tip = tip_speed(wsp150, turbine)*0.998
    # Water content in a unit volume of air [-]
    water_content = np.zeros(rain_rate.shape)
    # If the terminal velocity is zero, we cannot divide by it
    valid = np.where(v_term > 0)[0]
    water_content[valid] = np.divide(rain_rate_si[valid], v_term[valid])

    # We are going to run the model for every angle of impact and average the results at the end.
    damage_inc_tab = []
    rainfall_b_tab = []
    theta = np.linspace(0, 2*np.pi, 100)
    for t in theta:
        # Impact speed correction for the IEA and NREL turbines, we integrate for every angle of impact
        v_corrected = np.copy(v_tip)
        if turbine == 'IEA_15MW':
            # We only calculate the corrected impact speed for the valid indices 
            for i in valid:
                v_corrected[i] = ut.get_impact_speed(
                    v_tip[i] - v_term[i] * (1+np.sin(t)),
                    droplet_d[i]*1e-3 / 2,
                    ut.rc('iea', 0.998),
                    ut.n('iea', 0.998)
                )        

        elif turbine == 'NREL_5MW':
           for i in valid:
                v_corrected[i] = ut.get_impact_speed(
                    v_tip[i] - v_term[i] * (1+np.sin(t)),
                    droplet_d[i]*1e-3 / 2,
                    ut.rc('nrel', 0.998),
                    ut.n('nrel', 0.998)
                )        
        else:
            v_corrected = v_tip

        # Relative velocity of the rain drops
        v_rel = np.sqrt(wsp150**2 + v_corrected**2)
        
        # Define the rain rate depending on theta
        rain_rate_b = np.array(water_content * (v_tip + np.sin(t)*v_term)) 
        rainfall_b = rain_rate_b*dt

        # The droplet dependant c and m parameters
        if drop_dep == 'softsign':
            m_imp = -3.134*(droplet_d-2.107) / (1 + np.abs(droplet_d-2.107)) + 8.870
            imp_100 = -17.066*(droplet_d-2.294) / (1 + np.abs(droplet_d-2.294)) + 21.738
        elif drop_dep == 'linear':
            m_imp = -1.284 * droplet_d + 11.510
            imp_100 = -6.604 * droplet_d + 36.955
        c_imp = imp_100*100**m_imp

        # Impingement to damge as function of the relative velocity
        impingement= np.where(v_rel>0, c_imp*v_rel**(-m_imp), np.inf)
        # Damage increments
        damage_inc_tab.append(np.divide(rainfall_b,impingement))
        rainfall_b_tab.append(rainfall_b)

    # Average the results over all angles
    rainfall_b = np.array(rainfall_b_tab).mean(axis=0)
    damage_inc = np.array(damage_inc_tab).mean(axis=0)
    return rainfall_b, damage_inc

# %% Functions for impinged rain with speed correction and angle dependency separately
def impinged_rain_corr(wsp, rainfall, dt, turbine, droplet_d=None, v_term=None, drop_dep='softsign'):
    """A function that calculates impinged rain on a blade tip and corresponding damage increments taking into account the reduced terminal velocity of the droplets.
 
    Args:
        wsp (arra_type): Wind speed at hub height
        rainfall (array_type): The rainfall in [m]
        dt (int or float): The time step of the wsp and rainfall data [s]
        turbine (string):'NREL_5MW', 'DTU_10MW', or 'IEA_15MW'
 
    Returns:
        rainfall_b (array_like): The impinged rainfall on the blade tip [m]
        damage_inc (array_like): Tha damage increments. When they sum up to one damage begins to occur
    """
    # Wind profile power law
    wsp150 = wsp * (150/10)**(0.143)
    v_tip = tip_speed(wsp150, turbine)*0.99
    # Rain rate in SI units [m/s]
    rain_rate_si = rainfall/(1000*dt)
    # Rain rate in [mm/h]
    rain_rate = rain_rate_si*1000*3600
    if droplet_d is None:
        droplet_d = droplet_diameter(rain_rate)
    if v_term is None:
        v_term = terminal_velocity(droplet_d)
    # Water content in a unit volume of air [-]
    water_content = np.zeros(rain_rate.shape)
    valid = np.where(v_term > 0)[0]
    water_content[valid] = np.divide(rain_rate_si[valid], v_term[valid])
    # Rain rate on the blade tip [m/s] (SI unit)
    rain_rate_b = water_content * v_tip # [m/s]
    # Impinged rainfall on the blade tip [m]
    rainfall_b = rain_rate_b*dt
    # We calculate the corrected impact speed for the valid indices and for the right turbine model.
    v_corrected = np.copy(v_tip)
    if turbine == 'IEA_15MW':
        for i in valid:
            # ut.get_impact_speed is a function that calculates the impact speed of the droplets on the blade tip it takes 4 arguments: blade velocity, droplet radius, aerodynamic nose radius, and n a parameter of the blade.
            v_corrected[i] = ut.get_impact_speed(
                v_tip[i]-v_term[i],
                droplet_d[i]*1e-3 / 2,
                ut.rc('iea', 0.998),
                ut.n('iea', 0.998)
            )
    elif turbine == 'NREL_5MW':
        for i in valid:
            v_corrected[i] = ut.get_impact_speed(
                v_tip[i]*0.998,
                droplet_d[i]*1e-3 / 2,
                ut.rc('nrel', 0.998),
                ut.n('nrel', 0.998)
            )        
    else:
        v_corrected = v_tip*0.998
    # Average impact velocity of the rain drops relative to the blade (averaging out vertical velocity)
    v_rel = np.sqrt(wsp150**2 + v_corrected**2)
    # The droplet dependant c and m parameters
    if drop_dep == 'softsign':
        m_imp = -3.134*(droplet_d-2.107) / (1 + np.abs(droplet_d-2.107)) + 8.870
        imp_100 = -17.066*(droplet_d-2.294) / (1 + np.abs(droplet_d-2.294)) + 21.738
    elif drop_dep == 'linear':
        m_imp = -1.284 * droplet_d + 11.510
        imp_100 = -6.604 * droplet_d + 36.955
    c_imp = imp_100*100**m_imp
    # Impingement to damge as function of the relative velocity
    impingement = np.where(v_rel>0, c_imp*v_rel**(-m_imp), np.inf)
    # Damage increments
    damage_inc = np.divide(rainfall_b,impingement)
    return rainfall_b, damage_inc


def impinged_rain_theta(wsp, rainfall, dt, turbine, droplet_d=None, v_term=None, drop_dep='softsign'):
    """A function that calculates impinged rain on a blade tip and corresponding damage increments with angle dependency but WITHOUT speed correction.
 
    Args:
        wsp (arra_type): Wind speed at hub height
        rainfall (array_type): The rainfall in [m]
        dt (int or float): The time step of the wsp and rainfall data [s]
        turbine (string):'NREL_5MW', 'DTU_10MW', or 'IEA_15MW'
 
    Returns:
        rainfall_b (array_like): The impinged rainfall on the blade tip [m]
        damage_inc (array_like): Tha damage increments. When they sum up to one damage begins to occur
    """
    # Wind profile power law
    wsp150 = wsp * (150/10)**(0.143)
    # Rain rate in SI units [m/s]
    rain_rate_si = rainfall/(1000*dt)
    # Rain rate in [mm/h]
    rain_rate = rain_rate_si*1000*3600
    if droplet_d is None:
        droplet_d = droplet_diameter(rain_rate)
    if v_term is None:
        v_term = terminal_velocity(droplet_d)
    # Calculate the tip speed (the 0.998 factor is the fraction that defines where we are on the blade)
    v_tip = tip_speed(wsp150, turbine)*0.998
    # Water content in a unit volume of air [-]
    water_content = np.zeros(rain_rate.shape)
    valid = np.where(v_term > 0)[0]
    water_content[valid] = np.divide(rain_rate_si[valid], v_term[valid])

    # We are going to run the model for every angle of impact and average the results at the end.
    theta = np.linspace(0, 2*np.pi, 100)
    damage_inc_tab = []
    rainfall_b_tab = []
    for t in theta:
        # Impact speed correction for the IEA and NREL turbines, we integrate for every angle of impact
        v_rel = np.sqrt(wsp150**2 + (v_tip + v_term * (np.sin(t)))**2)
        
        # Define the rain rate depending on theta
        rain_rate_b = np.array(water_content * (v_tip + np.sin(t)*v_term)) 
        rainfall_b = rain_rate_b*dt

        # The droplet dependant c and m parameters
        if drop_dep == 'softsign':
            m_imp = -3.134*(droplet_d-2.107) / (1 + np.abs(droplet_d-2.107)) + 8.870
            imp_100 = -17.066*(droplet_d-2.294) / (1 + np.abs(droplet_d-2.294)) + 21.738
        elif drop_dep == 'linear':
            m_imp = -1.284 * droplet_d + 11.510
            imp_100 = -6.604 * droplet_d + 36.955
        c_imp = imp_100*100**m_imp

        # Impingement to damge as function of the relative velocity
        impingement= np.where(v_rel>0, c_imp*v_rel**(-m_imp), np.inf)
        # Damage increments
        damage_inc_tab.append(np.divide(rainfall_b,impingement))
        rainfall_b_tab.append(rainfall_b)

    # Average the results over all angles
    rainfall_b = np.array(rainfall_b_tab).mean(axis=0)
    damage_inc = np.array(damage_inc_tab).mean(axis=0)
    return rainfall_b, damage_inc


# %% Read in the ERA5 data
# Enter the windfarm name and the file path
# Example: AnholtHavn, Billund

WINDFARM = 'AnholtHavn'  # Change to 'Billund' if needed

if WINDFARM == 'Billund':
    file_path = 'droplet_impact/model_complete/erosion_data/meteorological_data/Billund.txt'

elif WINDFARM == 'AnholtHavn':
    file_path = 'droplet_impact/model_complete/erosion_data/meteorological_data/AnholtHavn.txt'

time_slice = slice('2010-01-01T00:00:00.000000000','2019-12-31T23:00:00.000000000')
hub_height = 150

##################### Test 1 with no correction #####################
print("Running Test 1: No correction")
# Read the data from the file
df = pd.read_csv(
    file_path,
    sep=r"\s+",
    comment="#",
    names=["year", "month", "day", "hour", "minute", "ws", "dir", "hyw", "hyT", "r10m"]
)

# Create datetime index
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
rainfall_b, damage_inc = ds_impinged_rain(ws_valid, rain_valid, dt, 'IEA_15MW')
df['rainfall_b'] = rainfall_b
df['damage_inc'] = damage_inc


# Save the results to a CSV file
if WINDFARM == 'AnholtHavn':
    df.to_csv('droplet_impact/model_complete/erosion_data/results/damage_increments_anholt_ini.csv')
elif WINDFARM == 'Billund':
    df.to_csv('droplet_impact/model_complete/erosion_data/results/damage_increments_billund_ini.csv')

print("Tests completed successfully.")
print("\n")

##################### Test 2 with only corrected speed #####################
print("----------------------")
print("Running Test 2: Corrected speed")
df = pd.read_csv(
    file_path,
    sep=r"\s+",
    comment="#",
    names=["year", "month", "day", "hour", "minute", "ws", "dir", "hyw", "hyT", "r10m"]
)

# Create datetime index
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

# Run the impinged rain function with corrected speed
rainfall_b, damage_inc = impinged_rain_corr(ws_valid, rain_valid, dt, 'IEA_15MW')
df['rainfall_b'] = rainfall_b
df['damage_inc'] = damage_inc

# Save the results to a CSV file
if WINDFARM == 'AnholtHavn':
    df.to_csv('droplet_impact/model_complete/erosion_data/results/damage_increments_anholt_corr.csv')
elif WINDFARM == 'Billund':
    df.to_csv('droplet_impact/model_complete/erosion_data/results/damage_increments_billund_corr.csv')


print("Tests completed successfully.")
print("\n")

##################### Test 3 with only angle effect #####################
print("----------------------")
print("Running Test 3: Angle effect")       
# Read the data from the file
df = pd.read_csv(
    file_path,
    sep=r"\s+",
    comment="#",
    names=["year", "month", "day", "hour", "minute", "ws", "dir", "hyw", "hyT", "r10m"]
)

# Create datetime index
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

# Run the impinged rain function with angle effect
# Note: This function does not include speed correction, only angle dependency
rainfall_b, damage_inc = impinged_rain_theta(ws_valid, rain_valid, dt, 'IEA_15MW')
df['rainfall_b'] = rainfall_b
df['damage_inc'] = damage_inc

# Save the results to a CSV file
if WINDFARM == 'AnholtHavn':
    df.to_csv('droplet_impact/model_complete/erosion_data/results/damage_increments_anholt_theta.csv')
elif WINDFARM == 'Billund':
    df.to_csv('droplet_impact/model_complete/erosion_data/results/damage_increments_billund_theta.csv')


print("Tests completed successfully.")
print("\n")

##################### Test 4 with the whole model #####################
print("----------------------")
print("Running Test 4: Angle effect and speed correction")
# This test combines both angle effect and speed correction in the impinged rain function
# Read the data from the file
df = pd.read_csv(
    file_path,
    sep=r"\s+",
    comment="#",
    names=["year", "month", "day", "hour", "minute", "ws", "dir", "hyw", "hyT", "r10m"]
)

# Create datetime index
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

# Run the impinged rain function with angle effect and speed correction
rainfall_b, damage_inc = ds_impinged_rain_theta(ws_valid, rain_valid, dt, 'IEA_15MW')
df['rainfall_b'] = rainfall_b
df['damage_inc'] = damage_inc

# Save the results to a CSV file
if WINDFARM == 'AnholtHavn':
    df.to_csv('droplet_impact/model_complete/erosion_data/results/damage_increments_anholt_theta_corr.csv')
elif WINDFARM == 'Billund':
    df.to_csv('droplet_impact/model_complete/erosion_data/results/damage_increments_billund_theta_corr.csv')

print("Tests completed successfully.")
print("\n")

