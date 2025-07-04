import numpy as np
from scipy.integrate import solve_ivp
import config as cfg
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import physics_model as pm
import visualization as vis


### In here we do 5 tests with different air velocities to see the effect on the forces acting on the droplet.
# The air velocity is varied from 50 m/s to 90 m/s in steps of 10 m/s.
# The initial conditions are the same for all tests, except for the air velocity.
# The results are printed in the console and can be plotted using the visualization module. 

# We can chose the plot to make at the end of this file.

# Solve the differential equations
time_span = (0, 0.1)  # Time span for the simulation
time_steps = np.linspace(time_span[0], time_span[1], 1000000)  # Time steps for evaluation


#--------------------------- TEST 1 ------------------------------
# # Initial conditions
initial_conditions = [cfg.x0, cfg.vx0, cfg.y0, cfg.vy0, cfg.a0, cfg.va0,-1,cfg.vx_blade0] # Initial conditions for the droplet
cfg.V_blade = 90   # Air velocity (m/s)
# Create an instance of the RaindropModel
model1 = pm.RaindropModel(initial_conditions)
events1 = [model1.hit_the_blade] 
sol1 = solve_ivp(model1.droplet_equations, time_span, initial_conditions, t_eval=time_steps, method='DOP853',rtol=1e-6, atol=1e-8,events=events1)
if sol1.t_events[0].size > 0:
    print("Simulation 1: Droplet hit the blade")
else:
    print("Simulation 1: Not long enough to hit the blade")

# Get the maximum forces
forces_table_1 = model1.get_forces_table()
print(forces_table_1)
print("--------------------------------------------------")

#--------------------------- TEST 2 ------------------------------
# Initial conditions
initial_conditions = [cfg.x0, cfg.vx0, cfg.y0, cfg.vy0, cfg.a0, cfg.va0,-1,cfg.vx_blade0] 
# Create an instance of the RaindropModel
model2 = pm.RaindropModel(initial_conditions)
events2 = [model2.hit_the_blade] 
cfg.V_blade = 80  # Air velocity (m/s)
sol2 = solve_ivp(model2.droplet_equations, time_span, initial_conditions, t_eval=time_steps, method='DOP853',rtol=1e-6, atol=1e-8,events=events2)
if sol2.t_events[0].size > 0:
    print("Simulation 2: Droplet hit the blade")
else:
    print("Simulation 2: Not long enough to hit the blade")

# Get the maximum forces
forces_table_2 = model2.get_forces_table()
print(forces_table_2)
print("--------------------------------------------------")


#--------------------------- TEST 3 ------------------------------

# Initial conditions
initial_conditions = [cfg.x0, cfg.vx0, cfg.y0, cfg.vy0, cfg.a0, cfg.va0,-1,cfg.vx_blade0] 
# Create an instance of the RaindropModel
model3 = pm.RaindropModel(initial_conditions)
events3 = [model3.hit_the_blade] 
cfg.V_blade = 70 # Air velocity (m/s)  
sol3 = solve_ivp(model3.droplet_equations, time_span, initial_conditions, t_eval=time_steps, method='DOP853',rtol=1e-6, atol=1e-8,events=events3)
if sol3.t_events[0].size > 0:
    print("Simulation 3: Droplet hit the blade")
else:
    print("Simulation 3: Not long enough to hit the blade")

# Get the maximum forces
forces_table_3 = model3.get_forces_table()
print(forces_table_3)
print("--------------------------------------------------")

#--------------------------- TEST 4 ------------------------------

# Initial conditions
initial_conditions = [cfg.x0, cfg.vx0, cfg.y0, cfg.vy0, cfg.a0, cfg.va0,-1,cfg.vx_blade0] 
# Create an instance of the RaindropModel
model4 = pm.RaindropModel(initial_conditions)
events4 = [model4.hit_the_blade] 
cfg.V_blade = 60 # Air velocity (m/s)  
sol4 = solve_ivp(model4.droplet_equations, time_span, initial_conditions, t_eval=time_steps, method='DOP853',rtol=1e-6, atol=1e-8,events=events4)
if sol4.t_events[0].size > 0:
    print("Simulation 4: Droplet hit the blade")
else:
    print("Simulation 4: Not long enough to hit the blade")

# Get the maximum forces
forces_table_4 = model4.get_forces_table()
print(forces_table_4)
print("--------------------------------------------------")

#--------------------------- TEST 5 ------------------------------

# Initial conditions
initial_conditions = [cfg.x0, cfg.vx0, cfg.y0, cfg.vy0, cfg.a0, cfg.va0,-1,cfg.vx_blade0] 
# Create an instance of the RaindropModel
model5 = pm.RaindropModel(initial_conditions)
events5 = [model5.hit_the_blade] 
cfg.V_blade = 50 # Air velocity (m/s)  
sol5 = solve_ivp(model5.droplet_equations, time_span, initial_conditions, t_eval=time_steps, method='DOP853',rtol=1e-6, atol=1e-8,events=events5)
if sol5.t_events[0].size > 0:
    print("Simulation 5: Droplet hit the blade")
else:
    print("Simulation 5: Not long enough to hit the blade")

# Get the maximum forces
forces_table_5 = model5.get_forces_table()
print(forces_table_5)
print("--------------------------------------------------")

# Plot the results
#vis.plot_all(sol1,model1)
#vis.plot_points_above_threshold(sol1,1.5)
#vis.plot_a_b(sol1)
#vis.plot_x_y(sol1)        
#vis.plot_mult_all([sol1, sol2, sol3,sol4,sol5],['$V_{blade} =$ 90 m/s', '$V_{blade} =$ 80 m/s', '$V_{blade} =$ 70 m/s', '$V_{blade} =$ 60 m/s', '$V_{blade} =$ 50 m/s'])              
#vis.plot_deltax_vair(sol1)
#vis.plot_deltax_deltav_mult([sol1, sol2, sol3,sol4,sol5],[90,80,70,60,50],['$V_{blade} =$ 90 m/s', '$V_{blade} =$ 80 m/s', '$V_{blade} =$ 70 m/s', '$V_{blade} =$ 60 m/s', '$V_{blade} =$ 50 m/s'])
#vis.plot_vair([sol1,sol2,sol3],[70,80,90],['$V_{blade} =$ 70 m/s', '$V_{blade} =$ 80 m/s', '$V_{blade} =$ 90 m/s'])
#vis.plot_CD(sol1)
#vis.plot_CD_mult([sol1, sol2, sol3,sol4,sol5],[90,80,70,60,50],['$V_{blade} =$ 90 m/s', '$V_{blade} =$ 80 m/s', '$V_{blade} =$ 70 m/s', '$V_{blade} =$ 60 m/s', '$V_{blade} =$ 50 m/s'])
#vis.plot_comparaison_deltav([sol1, sol2, sol3,sol4,sol5],[90,80,70,60,50],['$V_{blade} =$ 90 m/s', '$V_{blade} =$ 80 m/s', '$V_{blade} =$ 70 m/s', '$V_{blade} =$ 60 m/s', '$V_{blade} =$ 50 m/s'])


