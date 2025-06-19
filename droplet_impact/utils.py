import sys
import os
from scipy.integrate import solve_ivp
from scipy.stats import linregress
import pandas as pd
from openpyxl import load_workbook
import numpy as np
from . import config as cfg
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from . import physics_model as pm
import joblib
import importlib.resources


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

def load_interpolator():
    # Use the string name of your package and relative path
    with importlib.resources.files("droplet_impact.data").joinpath("impact_speed_interpolator.pkl").open("rb") as f:
        return joblib.load(f)

_interpolator = None

def get_impact_speed(V_blade, R, Rc, n):
    global _interpolator
    if _interpolator is None:
        _interpolator = load_interpolator()
    point = np.array([[V_blade, R, Rc, n]])
    return _interpolator(point)[0]

points_n_nrel = [
    [0.19858261842596042, 1.6543080939947783],
    [0.21589863574197776, 1.6543080939947783],
    [0.23192386377764967, 1.4506527415143604],
    [0.3427486351768336, 1.4496083550913839],
    [0.3751698258225673, 1.271018276762402],
    [0.4072067184338725, 1.2699738903394255],
    [0.4385518745832062, 1.1885117493472586],
    [0.5346580312637754, 1.187467362924282],
    [0.5659828422230512, 1.1154046997389035],
    [0.6620889989036204, 1.114360313315927],
    [0.6934002464028576, 1.0485639686684072],
    [0.9990279520305632, 1.0485639686684072]
]

points_n_iea = [
    [0.1983136098018604, 1.7785900783289819],
    [0.2035264996100505, 1.77023498694517],
    [0.21830162874549297, 1.7441253263707575],
    [0.24176189345902138, 1.7054830287206268],
    [0.25913668576853954, 1.6783289817232379],
    [0.2843240310603235, 1.6417754569190601],
    [0.30951589750545366, 1.6031331592689297],
    [0.33470324279723757, 1.566579634464752],
    [0.3650944355905192, 1.5258485639686685],
    [0.38072632328507, 1.5039164490861618],
    [0.39549014953714695, 1.483028720626632],
    [0.41198783809749867, 1.4610966057441255],
    [0.4302239101194715, 1.4360313315926894],
    [0.448455460988098, 1.4130548302872064],
    [0.4701547364732743, 1.3879895561357702],
    [0.4875159653227539, 1.3671018276762403],
    [0.5118216857120252, 1.3378590078328982],
    [0.5335254823505478, 1.3107049608355092],
    [0.6324098877623684, 1.2261096605744126],
    [0.7174301764380095, 1.1467362924281983],
    [0.759938060199157, 1.108093994778068],
    [0.779876346455981, 1.0966057441253263],
    [0.8474088139884486, 1.0966057441253263],
    [0.916672883252518, 1.0966057441253263],
    [0.999789766369401, 1.0966057441253263]
]

points_rc_iea = [
    [0.1989781588454875, 1.0072730982761422],
    [0.2371247614437007, 0.7986456658252736],
    [0.2501278301679196, 0.7427587894726373],
    [0.26312609775907114, 0.7059318448244676],
    [0.2769885693023553, 0.6757894187757365],
    [0.32465421839554137, 0.5474756806929585],
    [0.3497817484926442, 0.5017443060217907],
    [0.3731760695524144, 0.4631845938735316],
    [0.39917100435702824, 0.4214290711899564],
    [0.4234327301243094, 0.3862325833072669],
    [0.44683185231714684, 0.3488986177272438],
    [0.48583305660135795, 0.2909929178328557],
    [0.5092337791718846, 0.26097104315990055],
    [0.5326377024977895, 0.2306860582999048],
    [0.5577684333502707, 0.2083809882618082],
    [0.5820317594952409, 0.18960163302586247],
    [0.6062966860179003, 0.17127170299881303],
    [0.6314290172480707, 0.1535967135413591],
    [0.65569394377073, 0.1387475955951343],
    [0.6790946663412566, 0.12443294162639547],
    [0.7024953889117832, 0.11159513716532418],
    [0.7258993122376881, 0.09864482283703976],
    [0.7493048359412822, 0.0865690967734251],
    [0.7735729632193198, 0.07707713323138715],
    [0.7995614965131772, 0.0721867832471692],
    [0.8255452286739671, 0.06908935506098773],
    [0.8515305612124462, 0.06564840065088566],
    [0.8801149071180799, 0.06192647097859437],
    [0.9052360357044263, 0.05841921309163883],
    [0.9312245689982837, 0.054712661150182085],
    [0.9580805069996521, 0.05087128871219534],
    [0.9832064367190659, 0.0469602929839554],
    [0.9884092645864425, 0.0452884882871698],
    [0.9918804837941756, 0.04367756773035896],
    [0.9936312969860888, 0.04004604198427972],
    [0.9962847231946741, 0.031090226499526207],
    [0.9980995514941527, 0.02134555887174696],
    [0.9990741815068357, 0.01305422122296893]
]

points_rc_nrel = [
    [0.20082019356568148, 0.6215661483217255],
    [0.21640147074709631, 0.6203002537908704],
    [0.224243321423856, 0.49566381183840513],
    [0.23034076041945895, 0.4196715826523095],
    [0.23381838113794856, 0.39320445827625267],
    [0.2459396417554543, 0.3931183067879407],
    [0.2736453803097531, 0.39292145997455835],
    [0.3022185235715629, 0.38988901063746956],
    [0.32906165855141817, 0.3841045015553399],
    [0.3429145278285675, 0.3840083228563938],
    [0.3507275717069228, 0.34950530298188615],
    [0.3628936428997243, 0.2853818499646837],
    [0.3759239180446586, 0.23471052692770314],
    [0.4070976750513121, 0.22954431741838688],
    [0.42185155696744436, 0.19573178602368424],
    [0.4383402482985985, 0.16449840893787915],
    [0.46432238008169924, 0.15858261997434747],
    [0.49289872409888735, 0.15509934313451457],
    [0.5197450598341209, 0.1506043394658129],
    [0.5336011298666485, 0.14840476669031535],
    [0.5414125733673147, 0.13605090048469637],
    [0.5570386611240253, 0.11270097815525722],
    [0.5648517050023806, 0.10257483281485663],
    [0.5769777667529538, 0.10035160892667187],
    [0.6038241024881873, 0.09744327391267829],
    [0.6324020468830645, 0.09461626478753907],
    [0.6601173877034981, 0.09055358600136434],
    [0.6757402747048304, 0.07610493049212035],
    [0.6930979711211847, 0.06304134400197398],
    [0.7182142985744636, 0.06077516517824262],
    [0.7450606343096973, 0.059013812841597145],
    [0.7727743747524417, 0.056889742360386464],
    [0.7996255116207426, 0.05405553636322183],
    [0.8256076434038434, 0.05211155922994701],
    [0.8515897751869442, 0.050237492550789235],
    [0.8793051160073778, 0.04808037087921402],
    [0.9061562528756787, 0.045685041425469565],
    [0.9321447861695361, 0.04278644060512846],
    [0.9529576980167321, 0.036744625680977076],
    [0.97117799800753, 0.030880233481565957],
    [0.9876698900940624, 0.025579970696326945],
    [0.9980947503610854, 0.021813675242864136]
]

x_n_nrel = np.array(points_n_nrel)[:, 0]
y_n_nrel = np.array(points_n_nrel)[:, 1]
x_n_iea = np.array(points_n_iea)[:, 0]
y_n_iea = np.array(points_n_iea)[:, 1]
x_rc_iea = np.array(points_rc_iea)[:, 0]
y_rc_iea = np.array(points_rc_iea)[:, 1]
x_rc_nrel = np.array(points_rc_nrel)[:, 0]
y_rc_nrel = np.array(points_rc_nrel)[:, 1]

def calculate_value(r, points, is_log_scale=False):
    for i in range(len(points) - 1):
        x0, y0 = points[i]
        x1, y1 = points[i + 1]

        if x0 <= r <= x1:
            if is_log_scale:
                log_y0 = np.log(y0)
                log_y1 = np.log(y1)
                slope = (log_y1 - log_y0) / (x1 - x0)
                intercept = log_y0 - slope * x0
                return np.exp(slope * r + intercept)
            else:
                slope = (y1 - y0) / (x1 - x0)
                intercept = y0 - slope * x0
                return slope * r + intercept
    return None

def n(model, r):
    if model == 'nrel':
        return calculate_value(r, points_n_nrel)
    elif model == 'iea':
        return calculate_value(r, points_n_iea)
    else:
        raise ValueError("Invalid model. Choose either 'nrel' or 'iea'.")

def rc(model, r):
    if model == 'nrel':
        return calculate_value(r, points_rc_nrel, is_log_scale=True)
    elif model == 'iea':
        return calculate_value(r, points_rc_iea, is_log_scale=True)
    else:
        raise ValueError("Invalid model. Choose either 'nrel' or 'iea'.")

