import sys
import os
from scipy.integrate import solve_ivp
from scipy.stats import linregress
import pandas as pd
from openpyxl import load_workbook
import numpy as np
import config as cfg
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import physics_model as pm
import visualization as vis

def extract_data(feuille, x_range, y_range):
    x_data = []
    y_data = []
    for row in feuille[x_range]:
        for cell in row:
            x_data.append(cell.value)
    for row in feuille[y_range]:
        for cell in row:
            y_data.append(cell.value)
    return np.array(x_data), np.array(y_data)

def linear_regression(x, y):
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    return slope, intercept, r_value, p_value, std_err

def corrected_speed(x,vv,diameter, nose_radius, n):
    """
    Calculate the reduced speed of the blade. Configuration RET.
    """
    time_span = (0, 0.1)  # Time span for the simulation
    time_steps = np.linspace(time_span[0], time_span[1], 100000)  # Time steps for evaluation
    cfg.R = diameter / 2
    cfg.Rc_alpha = nose_radius
    cfg.n = n
    res = []
    for v in vv:
        cfg.V_blade = v
        initial_conditions = [0, 0, 0, -v_terminal(diameter/2 *1e3), diameter/2, 0,-0.2,v] # Initial conditions for the droplet
        mod = pm.RaindropModel(initial_conditions)
        events = [mod.hit_the_blade]
        sol = solve_ivp(mod.droplet_equations, time_span, initial_conditions, t_eval=time_steps, method='RK45',events=events)
        res.append(v-sol.y[1,-1])
    return np.array(res)

def impact_speed_vertical(initial_conditions, time_span, time_steps, nose_radius, n, initial_radius, blade_speed):
    """
    Calculate the impact speed of the droplet on the blade. Configuration real turbine.
    """
    cfg.R = initial_radius
    cfg.Rc_alpha = nose_radius
    cfg.n = n
    cfg.V_blade = blade_speed
    mod = pm.RaindropModel(initial_conditions)
    events = [mod.hit_the_blade_vertical]
    sol = solve_ivp(mod.droplet_equations_vertical, time_span, initial_conditions, t_eval=time_steps, method='DOP853',events=events, rtol=1e-6, atol=1e-8)
    return blade_speed-sol.y[1,-1]


A = [-8.5731540e-2, 3.3265862, 4.3843578, -6.8813414, 4.7570205, -1.9046601, 4.6339978e-1,-6.7607898e-2, 5.4455480e-3,-1.8631087e-4]
def v_terminal(r):
    """
    Calculate the terminal velocity of a raindrop based on its radius and the given coefficients.
    """
    d = 2*r
    return A[0] + A[1]*d + A[2]*d**2 + A[3]*d**3 + A[4]*d**4 + A[5]*d**5 + A[6]*d**6 + A[7]*d**7 + A[8]*d**8 + A[9]*d**9