import matplotlib.pyplot as plt
from . import config as cfg
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

def plot_all(sol,mod):
    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 4, 1)
    plt.plot(sol.t, sol.y[0], label='x(t) (m)')
    plt.legend()
    plt.grid()

    plt.subplot(2, 4, 5)
    plt.plot(sol.t, sol.y[1], label='vx(t) (m/s)')
    plt.legend()
    plt.grid()

    plt.subplot(2, 4, 2)
    plt.plot(sol.t, sol.y[2], label='y(t) (m)')
    plt.legend()
    plt.grid()

    plt.subplot(2, 4, 6)
    plt.plot(sol.t, sol.y[3], label='vy(t) (m/s)')
    plt.legend()
    plt.grid()

    plt.subplot(2, 4, 3)
    plt.plot(sol.t, sol.y[4], label='a(t) (m)')
    plt.legend()
    plt.grid()

    plt.subplot(2, 4, 7)
    plt.plot(sol.t, sol.y[5], label='va(t) (m/s)')
    plt.legend()
    plt.grid()

    # Calculate forces
    forces_table = mod.get_forces_table()
    F_p = forces_table['F_p']
    F_s = forces_table['F_s']
    T = forces_table['time']
    plt.subplot(2, 4, 4)
    plt.plot(T, F_p, label='F_p(t) (N)')
    plt.plot(T, F_s, label='F_s(t) (N)')
    plt.plot(T, F_p + F_s, label='F_p + F_s (N)')
    plt.legend()
    plt.grid()

    plt.subplot(2, 4, 8)
    plt.plot(sol.t, sol.y[7], label='vx_blade(t) (m/s)')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

def plot_a_b(sol):
    b = cfg.R**3 / (sol.y[4]**2)
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plt.plot(sol.t, sol.y[4], label='a(t) (m)')
    plt.legend()
    plt.grid()

    plt.subplot(2, 2, 2)
    plt.plot(sol.t, b, label='b(t) (m)')
    plt.legend()
    plt.grid()

    epsilon = np.sqrt(1-(b/sol.y[4])**2)
    plt.subplot(2, 2, 3)
    plt.plot(sol.t, epsilon, label='epsilon(t)')
    plt.legend()
    plt.grid()

    plt.subplot(2, 2, 4)
    plt.plot(sol.t, sol.y[4]/b, label='a/b')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

def plot_x_y(sol):
    plt.figure(figsize=(6, 6))
    plt.plot(sol.y[0], sol.y[2], label='Trajectory (m)')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Droplet Trajectory')
    plt.legend()
    plt.grid()
    plt.axis('equal')
    plt.show()

# I want to do the same plot as plot_all but for every sol in sols they must be in the same figure bur different colors

def plot_mult_all(sols, cfgs):
    # Plot the results
    plt.figure(figsize=(12, 6))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i, sol in enumerate(sols):
        plt.subplot(2, 3, 1)
        plt.plot(sol.t, sol.y[0], label=cfgs[i], color=colors[i])
        plt.title('x(t) (m)')
        plt.legend()
        plt.tight_layout()
        plt.grid()

        plt.subplot(2, 3, 4)
        plt.plot(sol.t, sol.y[1], label=cfgs[i], color=colors[i])
        plt.title('vx(t) (m/s)')
        plt.legend()
        plt.tight_layout()
        plt.grid()

        plt.subplot(2, 3, 2)
        plt.plot(sol.t, sol.y[2], label=cfgs[i], color=colors[i])
        plt.title('y(t) (m)')
        plt.legend()
        plt.tight_layout()
        plt.grid()

        plt.subplot(2, 3, 5)
        plt.plot(sol.t, sol.y[3], label=cfgs[i], color=colors[i])
        plt.title('vy(t) (m/s)')
        plt.legend()
        plt.tight_layout()
        plt.grid()

        plt.subplot(2, 3, 3)
        plt.plot(sol.t, sol.y[4],label=cfgs[i], color=colors[i])
        plt.title('a(t) (m)')
        plt.legend()
        plt.tight_layout()
        plt.grid()

        plt.subplot(2, 3, 6)
        plt.plot(sol.t, sol.y[5], label=cfgs[i], color=colors[i])
        plt.title('va(t) (m/s)')
        plt.legend()
        plt.tight_layout()
        plt.grid()


    plt.tight_layout()
    plt.show()

def plot_deltax_vair(sol):
    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(sol.y[6]-sol.y[0], cfg.V_blade*(1/pow(1+abs(sol.y[6]-sol.y[0])/cfg.Rc_alpha,cfg.n)), label='$V_{air} = f(\Delta x) $ (m/s)')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

def plot_deltax_deltav_mult(sols, param, cfgs):
    # Plot the results
    plt.figure(figsize=(10, 8))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i, sol in enumerate(sols):
        plt.plot(sol.y[6]-sol.y[0],param[i] - sol.y[1], label=cfgs[i], color=colors[i])
        plt.legend()
        plt.grid()

    plt.xlabel('$\Delta x$(m)')
    plt.ylabel('$\Delta v $(m/s)')
    plt.ylim(40,95)
    plt.title('$\Delta v = f(\Delta x)$')
    plt.tight_layout()
    plt.show()

# I want to plot v_air as a function of time
def plot_vair(sols, param, cfgs):
    # Plot the results
    plt.figure(figsize=(10, 8))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i, sol in enumerate(sols):
        plt.plot(sol.t, param[i]/pow(1+abs(sol.y[6]-sol.y[0])/cfg.Rc_alpha,cfg.n), label=cfgs[i], color=colors[i])
        plt.legend()
        plt.grid()

    plt.xlabel('t(s)')
    plt.ylabel('$V_{air} $(m/s)')
    plt.title('$V_{air} = f(t)$')
    plt.tight_layout()
    plt.show()

# I want to plot C_d, C_dyn and C_stat as a function of time

def plot_CD(sol):
    # Compute C_stat and C_dyn
    C_D_sphere = 0.4383
    C_stat = pow(C_D_sphere,(cfg.R**3)/(sol.y[4])) * pow(cfg.C_D_disk,1 - (cfg.R**3)/(sol.y[4]))
    b = cfg.R**3 / (sol.y[4]**2)
    V_slip_x = cfg.V_blade/pow(1+abs(sol.y[6]-sol.y[0])/cfg.Rc_alpha,cfg.n) - sol.y[1]
    F_drag_stat_x = 0.5 * cfg.rho_air * V_slip_x**2 * np.pi * sol.y[4]**2 * C_stat  + 0.5*cfg.rho_air*np.pi*sol.y[4]**2*cfg.k*b*(cfg.n*cfg.V_blade*(cfg.V_blade-sol.y[1])/(cfg.Rc_alpha*pow(1+abs(sol.y[0]-sol.y[6])/cfg.Rc_alpha,cfg.n+1)))  # Drag force
    dvx_dt = F_drag_stat_x / (4/3 * np.pi * cfg.R**3 * cfg.rho_water+0.5*cfg.rho_air*np.pi*cfg.R**3*cfg.k)
    C_dyn = cfg.k * b / (V_slip_x**2) * (-cfg.n*cfg.V_blade*(sol.y[1]-cfg.V_blade)/(cfg.Rc_alpha*pow(1+abs(sol.y[0]-sol.y[6])/cfg.Rc_alpha,cfg.n+1)) - dvx_dt)
    C_dyn_1 = -cfg.k * b* (cfg.n*cfg.V_blade*(sol.y[1]-cfg.V_blade)/(cfg.Rc_alpha*pow(1+abs(sol.y[0]-sol.y[6])/cfg.Rc_alpha,cfg.n+1))) / (V_slip_x**2)
    C_dyn_2 = cfg.k * b * dvx_dt / (V_slip_x**2)

    # Plot the results
    plt.figure(figsize=(10, 8))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    plt.plot(sol.t, C_stat + C_dyn, label='$C_D$', color=colors[2],linewidth=3)
    plt.plot(sol.t, C_stat, label='$C_{stat}$', color=colors[0])
    plt.plot(sol.t, C_dyn_1, label='$C_{dyn_1}$', color=colors[1])
    plt.plot(sol.t, C_dyn_2, label='$C_{dyn_2}$', color=colors[4])
    plt.plot(sol.t, C_dyn, label='$C_{dyn} = C_{dyn_1} - C_{dyn_2}$', color=colors[3])
    plt.legend()
    plt.grid()
    plt.xlabel('t(s)')
    plt.ylabel('C_d')
    plt.title('$C_D$, $C_{stat}$ and $C_{dyn}$ as a function of time')
    plt.tight_layout()
    plt.show()

def plot_CD_mult(sols, param, cfgs):
    # Compute C_stat and C_dyn
    C_D_sphere = 0.4383
    plt.figure(figsize=(10, 8))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i, sol in enumerate(sols):
        C_stat = pow(C_D_sphere,(cfg.R**3)/(sol.y[4])) * pow(cfg.C_D_disk,1 - (cfg.R**3)/(sol.y[4]))
        b = cfg.R**3 / (sol.y[4]**2)
        V_slip_x = cfg.V_blade/pow(1+abs(sol.y[6]-sol.y[0])/cfg.Rc_alpha,cfg.n) - sol.y[1]
        F_drag_stat_x = 0.5 * cfg.rho_air * V_slip_x**2 * np.pi * sol.y[4]**2 * C_stat  + 0.5*cfg.rho_air*np.pi*sol.y[4]**2*cfg.k*b*(cfg.n*cfg.V_blade*(cfg.V_blade-sol.y[1])/(cfg.Rc_alpha*pow(1+abs(sol.y[0]-sol.y[6])/cfg.Rc_alpha,cfg.n+1)))  # Drag force
        dvx_dt = F_drag_stat_x / (4/3 * np.pi * cfg.R**3 * cfg.rho_water+0.5*cfg.rho_air*np.pi*cfg.R**3*cfg.k)
        C_dyn = cfg.k * b / (V_slip_x**2) * (-cfg.n*cfg.V_blade*(sol.y[1]-cfg.V_blade)/(cfg.Rc_alpha*pow(1+abs(sol.y[0]-sol.y[6])/cfg.Rc_alpha,cfg.n+1)) - dvx_dt)
        plt.plot(sol.t, C_stat + C_dyn, label='$C_D$ for ' + cfgs[i], color=colors[i],linewidth=1)
    plt.legend()
    plt.grid()
    plt.xlabel('t(s)')
    plt.ylabel('C_D')
    plt.title('$C_D$ as a function of time')
    plt.tight_layout()
    plt.show()

def plot_points_above_threshold(sol, threshold=1.5):
    x = sol.y[0]
    vx = sol.y[1]
    start_index = next((i for i, v in enumerate(vx) if v > threshold), None)

    if start_index is not None:
        x_above_threshold = x[start_index:]
        vx_above_threshold = vx[start_index:]
        t_above_threshold = sol.t[start_index:]
        t2 = (t_above_threshold-min(t_above_threshold))*1e6
        x2 = x_above_threshold*1e6

        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(t2,x2)
        plt.xlabel('$t(\mu s)$')
        plt.xlim(min(t2), max(t2))
        plt.ylabel('$x(t) (\mu m)$')
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(t2,vx_above_threshold)
        plt.xlabel('$t(\mu s)$')
        plt.xlim(min(t2), max(t2))
        plt.ylabel('vx(t) (m/s)')
        plt.grid(True)
        plt.show()
        plt.tight_layout()
    else:
        print("Aucun point ne dépasse le seuil donné.")

#-------------------------####### Data for the plot comparaison_deltav #######--------------------
import os,sys

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, '..', 'data', 'inputs')

sys.path.append(data_dir)

# Importer le module (remplace 'data_vargas_2011' par le nom réel du fichier sans l'extension .py)
from . import data_vargas_2011 as data

def plot_comparaison_deltav(sols, param, cfgs):
    # Plot the results
    plt.figure(figsize=(10, 8))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    symbols = ['o', 's', 'D', '^', 'v', 'x', '+']
    patch=[]
    for i, sol in enumerate(sols):
        plt.plot(sol.y[6]-sol.y[0],param[i] - sol.y[1], color=colors[i])
        plt.plot(data.X[i], data.Y[i], symbols[i], color=colors[i])
        patch_i = plt.Line2D([0], [0], color=colors[i], lw=4, label=cfgs[i])
        patch.append(patch_i)
        plt.grid()

    handles, labels = plt.gca().get_legend_handles_labels()
    handles.extend(patch)
    labels.extend(cfgs)


    plt.xlabel('$\Delta x$(m)')
    plt.ylabel('$\Delta v $(m/s)')
    plt.ylim(40,95)
    plt.title('Comparison between our model and the data from Vargas et al. 2011')
    plt.legend(handles=handles, labels=labels, loc = 'upper left')
    plt.tight_layout()
    plt.show()