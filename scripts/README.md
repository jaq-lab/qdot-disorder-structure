# Scripts Directory

This directory contains Python analysis scripts for the Statistical Structure of Charge Disorder in Si/SiGe Quantum Dots project.

## Contents

### `Definitions.py`
Core analysis functions and utilities for quantum dot parameter extraction and visualization.

#### Key Functions

**Data Loading and Processing:**
- `load_data(path)` - Load simulation data from CSV files
- `get_data_sweep(data, ien=0)` - Extract energy sweep data
- `get_data_map(data)` - Convert data to 2D map format
- `get_xy(data)` - Extract x,y coordinate labels

**Visualization:**
- `plot_sweep(ax, detuning, energy, title, ...)` - Plot energy vs detuning sweeps
- `plot_map(ax, data_map, xlabels, ylabels, title, ...)` - Plot 2D parameter maps
- `draw_gates(xlabels, ylabels)` - Draw quantum dot gate structures
- `draw_gates2(xlabels, ylabels, ax, lw)` - Alternative gate drawing function

**Parameter Extraction:**
- `get_Eav_tc_eps0(eps, E)` - Calculate average energy, tunnel coupling, and detuning
- `calculate_lever_arm(det_voltage, Energy, i=18)` - Calculate lever arm parameter
- `get_fit(x, y)` - Fit avoided crossing data to extract parameters

**Statistical Analysis:**
- `get_DQDparams(file_paths, _plot_=True, _fit_=True)` - Extract parameters from multiple files
- `calculate_error(a, b)` - Calculate element-wise error between arrays

**Utility Functions:**
- `extract_number_from_name(name)` - Extract numeric values from filenames
- `get_subdir_paths(data_root)` - Get sorted subdirectory paths
- `coord_real2plot(xlabels, ylabels, xy)` - Convert real coordinates to plot coordinates

## Usage Example

```python
from scripts.Definitions import load_data, plot_sweep, get_Eav_tc_eps0
import matplotlib.pyplot as plt

# Load simulation data
data = load_data('path/to/simulation_data.csv')

# Extract detuning and energy data
detuning = data[:, 0]
energy = data[:, 1:]

# Calculate quantum dot parameters
E_avg, tc, eps0 = get_Eav_tc_eps0(detuning, energy)

# Plot results
fig, ax = plt.subplots()
plot_sweep(ax, detuning, energy[:, 0], "Energy vs Detuning")
plt.show()
```

## Dependencies

The scripts require the following Python packages:
- numpy
- matplotlib
- scipy
- pandas
- glob
- os
- re

## Citation

If you use these scripts in your research, please cite our paper:
```bibtex
@article{samadi2024statistical,
  title={Statistical Structure of Charge Disorder in Si/SiGe Quantum Dots},
  author={Samadi, Saeed and Cywi{\'n}ski, {\L}ukasz and Krzywda, Jan A},
  journal={arXiv preprint arXiv:2510.13578},
  year={2024}
}
```
