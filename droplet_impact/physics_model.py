from . import config as cfg
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import pandas as pd

class RaindropModel:
    def __init__(self, initial_conditions):
        self.initial_conditions = initial_conditions
        self.forces = {
            'time': [],
            'F_drag_stat_x': [],
            'F_v': [],
            'F_p': [],
            'F_s': []
        }

    def droplet_equations(self, t, z):
        x, vx, y, vy, a, va, x_blade, vx_blade = z

        b = (cfg.R**3) / (a**2)  # Semi-minor axis of the droplet

        # Relative velocity (slip velocity)
        vx_blade = cfg.V_blade 
        V_air = cfg.V_blade/pow(1+abs(x_blade-x)/cfg.Rc_alpha,cfg.n)
        V_slip_x = V_air - vx
        V_slip_y = vy

        # Calculate Reynolds number
        Re = (2 * a * cfg.rho_air * abs(V_slip_x)) / cfg.mu_air

        # Calculate drag coefficient
        if Re<=1:
            C_D_sphere = 27.6
        elif Re<=1000:
            C_D_sphere = 24/Re*(1 + 0.15*Re**0.687)
        else:
            C_D_sphere = 0.4383
            
        C_stat = pow(C_D_sphere,b/a) * pow(cfg.C_D_disk,1 - b/a)

        # Calculate Weber number
        We = (cfg.rho_air * vx_blade**2 * 2 * a) / cfg.sigma_water

        a_max = cfg.R * min(2.2, 3.4966 * pow(We,-0.1391))  # Maximum radius of the droplet
        bool = True
        if a > a_max:
            a = a_max
            b = (cfg.R**3) / (a**2)  # Update semi-minor axis of the droplet    
            da_dt = 0
            dva_dt = 0
            bool = False
        
        # Forces acting on the droplet
        F_drag_stat_x = 0.5 * cfg.rho_air * V_slip_x**2 * np.pi * a**2 * C_stat  + 0.5*cfg.rho_air*np.pi*a**2*cfg.k*b*(cfg.n*vx_blade*(cfg.V_blade-vx)/(cfg.Rc_alpha*pow(1+abs(x-x_blade)/cfg.Rc_alpha,cfg.n+1)))  # Drag force
        F_v = - (64/9) * cfg.mu_air * np.pi*cfg.R**3 *va / a**2   # Viscous force
        F_p = 0.5 * cfg.rho_air * V_slip_x**2 * cfg.C_p * np.pi * cfg.R**2  # Pressure force

        # Surface tension force
        if a > cfg.R:
            epsilon = np.sqrt(1-(b/a)**2)
        
            F_s = -(4/3) * cfg.sigma_water * np.pi*(4*a + 2*cfg.R**6/((a**5)*epsilon)*((3/epsilon)-np.arctanh(epsilon)*(1 + 3/(epsilon**2))))  # Surface tension force"
       
        else:
            F_s = 0
        # Store forces values
        self.forces['time'].append(t)
        self.forces['F_drag_stat_x'].append(F_drag_stat_x)
        self.forces['F_v'].append(F_v)
        self.forces['F_p'].append(F_p)
        self.forces['F_s'].append(F_s)

        # Differential equations
        dx_dt = vx
        dvx_dt = F_drag_stat_x / (4/3 * np.pi * cfg.R**3 * cfg.rho_water+0.5*cfg.rho_air*np.pi*cfg.R**3*cfg.k)
        dy_dt = vy
        dvy_dt = -cfg.g - 0.5*cfg.rho_air*vx*vy*np.pi*(a**2)*(C_stat+cfg.k*b*(cfg.n*cfg.V_blade*(cfg.V_blade-vx)/(cfg.Rc_alpha*pow(1+abs(x-x_blade)/cfg.Rc_alpha,cfg.n+1))-dvx_dt)/(V_slip_x**2))/(4/3 * np.pi * cfg.R**3 * cfg.rho_water)
        if bool: 
            da_dt = va
            dva_dt = (16/3) * (F_v + F_p + F_s) / (4/3 * np.pi * cfg.R**3 * cfg.rho_water)
        dxblade_dt = vx_blade

        return [dx_dt, dvx_dt, dy_dt, dvy_dt, da_dt, dva_dt, dxblade_dt, 0] 
    
    def hit_the_blade(self,t,z):
        return z[0] - z[6] - 1e-12  # Check if the droplet has hit the blade
    hit_the_blade.terminal = True

    def droplet_breakup(self,t,z):
        We = (cfg.rho_air * cfg.V_blade**2 * 2 * z[4]) / cfg.sigma_water
        return z[4] - cfg.R * min(2.2, 3.4966 * We)
    droplet_breakup.terminal = True  

    def get_forces_table(self):
        return pd.DataFrame(self.forces)

    def b(self, t, z):
        a = z[4]
        return 1 - (cfg.R**3 / (a**2))**2 / a**2 - 1e-12  # Semi-minor axis of the droplet
    b.terminal = True


    def droplet_equations_vertical(self, t, z):
        x, vx, a, va, x_blade, vx_blade = z

        b = (cfg.R**3) / (a**2)  # Semi-minor axis of the droplet

        # Relative velocity (slip velocity)
        vx_blade = cfg.V_blade
        V_air = cfg.V_blade/pow(1+abs(x_blade-x)/cfg.Rc_alpha,cfg.n)
        V_slip_x = V_air - vx

        # Calculate Reynolds number
        Re = (2 * a * cfg.rho_air * abs(V_slip_x)) / cfg.mu_air

        # Calculate drag coefficient
        if Re<=1:
            C_D_sphere = 27.6
        elif Re<=1000:
            C_D_sphere = 24/Re*(1 + 0.15*Re**0.687)
        else:
            C_D_sphere = 0.4383
            
        C_stat = pow(C_D_sphere,b/a) * pow(cfg.C_D_disk,1 - b/a)

        # Calculate Weber number
        We = (cfg.rho_air * vx_blade**2 * 2 * a) / cfg.sigma_water

        a_max = cfg.R * min(2.2, 3.4966 * We)  # Maximum radius of the droplet
        bool = True
        if a > a_max:
            a = a_max
            b = (cfg.R**3) / (a**2)  # Update semi-minor axis of the droplet    
            da_dt = 0
            dva_dt = 0
            bool = False
        
        # Forces acting on the droplet
        F_drag_stat_x = 0.5 * cfg.rho_air * V_slip_x**2 * np.pi * a**2 * C_stat  + 0.5*cfg.rho_air*np.pi*a**2*cfg.k*b*(cfg.n*vx_blade*(cfg.V_blade-vx)/(cfg.Rc_alpha*pow(1+abs(x-x_blade)/cfg.Rc_alpha,cfg.n+1)))  # Drag force
        F_v = - (64/9) * cfg.mu_air * np.pi*cfg.R**3 *va / a**2   # Viscous force
        F_p = 0.5 * cfg.rho_air * V_slip_x**2 * cfg.C_p * np.pi * cfg.R**2  # Pressure force

        # Surface tension force
        if a > cfg.R:
            epsilon = np.sqrt(1-(b/a)**2)
        
            F_s = -(4/3) * cfg.sigma_water * np.pi*(4*a + 2*cfg.R**6/((a**5)*epsilon)*((3/epsilon)-np.arctanh(epsilon)*(1 + 3/(epsilon**2))))  # Surface tension force"
       
        else:
            F_s = 0
        # Store forces values
        self.forces['time'].append(t)
        self.forces['F_drag_stat_x'].append(F_drag_stat_x)
        self.forces['F_v'].append(F_v)
        self.forces['F_p'].append(F_p)
        self.forces['F_s'].append(F_s)

        # Differential equations
        dx_dt = vx
        dvx_dt = -cfg.g + F_drag_stat_x / (4/3 * np.pi * cfg.R**3 * cfg.rho_water+0.5*cfg.rho_air*np.pi*cfg.R**3*cfg.k)
        if bool:
            da_dt = va
            dva_dt = (16/3) * (F_v + F_p + F_s) / (4/3 * np.pi * cfg.R**3 * cfg.rho_water)
        dxblade_dt = vx_blade

        return [dx_dt, dvx_dt, da_dt, dva_dt, dxblade_dt, 0] 
    
    def hit_the_blade_vertical(self,t,z):
        return z[0] - z[4] - 1e-12
    hit_the_blade_vertical.terminal = True

