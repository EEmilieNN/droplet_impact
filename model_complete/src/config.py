import numpy as np

# Physic parameters
# Constants
rho_air = 1.225  # kg/m^3
rho_water = 999.1  # kg/m^3
sigma_water = 0.07349  # N/m
mu_air = 1.7965e-5  # Pa·s
C_D = 0.47  # Drag coefficient (example)
C_p = 1.17  # Pressure coefficient (example)
g = 9.81  # m/s^2 (acceleration due to gravity)

# Simulation parameters
V_blade = 90  # Air velocity (m/s)
R = 0.49e-3  # Radius of the droplet (in meters)
C_D_disk = 1.17 # Drag coefficient for a disk
k = 9  # Coefficient for the drag force
alpha = 0 # Angle of attack of the droplet (in degrees)
Rc_alpha =  0.071 # Radius of the cylinder modeling the b    lade (in meters)
n=1.3 # Exponent for the velocity profile

# Initial conditions
x0 = 0.0  # Initial position
vx0 = 0.0  # Initial horizontal velocity
y0 = 0.0  # Initial height
vy0 = -2.0  # Initial vertical velocity
a0 = R # Initial radius (in meters)
va0 = 0.0  # Initial expansion velocity
x_blade0 = -0.08  # Initial position of the blade
vx_blade0 = V_blade  # Initial velocity of the blade