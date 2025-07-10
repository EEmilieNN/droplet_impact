
# Rain Droplet Impact on Wind Turbine Blade

This repository provides a set of Python tools to model, analyze, and visualize the impact of rain droplets on wind turbine blades. It includes modules for configuration, physical modeling, experimental data interpolation, and results visualization.

## Project Structure

- `droplet_impact/`: main source code for the package to import.
  - `config.py`: model configuration parameters.
  - `physics_model.py`: physical modeling functions for droplet impact.
  - `utils.py`: utility functions such as : get_impact_speed(V_blade, R_droplet, Rc, n) returning V_impact, V_terminal(R_droplet) returning the fall speed of the droplet, n(model,r/R_blade) and Rc(model, r/R_blade) returning the geometry parameters of the two turbines : NREL_5MW and IEA_15MW.
  - `data/`: experimental data and interpolation files.

- `model_complete`: whole model to clone if needed.
  - `data`: data used in the droplet model.
  - `erosion_data`: era5 input data and results in .csv files.
  - `tip_speed`: tables to convert wsp to tip speed.
  - `src`: python scripts to run the model.
     - `erosion`: scripts linked to erosion
         - `calculate_erosion_final.py`: run 4 simulations for different configurations after chosing the input data.
         - `visual_erosion_time.py`: plot the results from previous script.
     - `droplet_impact`: basically the code for the package.
  - `notebooks`: different notebooks to visualize the results.
     - `turbine_comp.ipynb`: shows the turbine comparison between IEA 15MW and NREL 5MW.
     - `RETanalysis.ipynb`: shows the impact of droplets speed reduction in the RET.
     - `computed_results.ipynb`: creates the interpolator that gives us get_impact_speed.
     - `results.ipynb`, `vertical_tests.ipynb`.
  - `plots`: interesting plots for the paper


## Installation

1. Import the package
```bash
pip install git+https://github.com/EEmilieNN/droplet_impact.git
```

2. To modify or use the model locally, you can clone this repository:
```bash
git clone https://github.com/EEmilieNN/droplet_impact.git
cd droplet_impact
```

## Usage

Example of importing and using the model:

```python
import droplet_impact.utils as ut
V = ut.get_impact_speed(V_blade,R,Rc,n)
```
# Use the module's functions as needed


## References

Scientific articles and experimental data are provided in the `Documentation/` folder for further understanding of the phenomenon.

## Author

Emilien Gouffault (emilien.gouffault@polytechnique.org)