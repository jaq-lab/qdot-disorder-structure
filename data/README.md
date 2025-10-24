# Data Directory

This directory contains simulation data and covariance matrices for the Statistical Structure of Charge Disorder in Si/SiGe Quantum Dots project.

## Contents

### Covariance Matrices

The repository contains three types of covariance matrices for two different charge densities:

#### Charge Density: 1×10¹⁰ cm⁻² (`1e10`)
- `cov_matrix_raw_1e10.npy` - Original simulation results
- `cov_matrix_pca_1e10.npy` - Principal component analysis results
- `cov_matrix_artificial_1e10.npy` - Generated using multivariate normal distribution

#### Charge Density: 5×10⁹ cm⁻² (`5e9`)
- `cov_matrix_raw_5e9.npy` - Original simulation results
- `cov_matrix_pca_5e9.npy` - Principal component analysis results
- `cov_matrix_artificial_5e9.npy` - Generated using multivariate normal distribution

## Data Description

### Parameters Analyzed
The covariance matrices contain correlations between the following quantum dot parameters:
- `d` - Detuning parameter
- `tcs` - Tunnel coupling strength
- `Lxavg` - Average dot size in x-direction
- `dLx` - Variation in dot size
- `Favg` - Average electric field
- `dF` - Electric field variation
- `eps` - Energy parameter
- `EL_orb` - Left orbital energy
- `ER_orb` - Right orbital energy
- `V_acs` - AC Stark voltage
- `Bhs` - Magnetic field strength
- `ratioL` - Left dot ratio
- `ratioR` - Right dot ratio

### Usage
These matrices can be loaded in Python using:
```python
import numpy as np
cov_matrix = np.load('cov_matrix_raw_1e10.npy', allow_pickle=True)
```

## Citation
If you use this data in your research, please cite our paper:
```bibtex
@article{samadi2024statistical,
  title={Statistical Structure of Charge Disorder in Si/SiGe Quantum Dots},
  author={Samadi, Saeed and Cywi{\'n}ski, {\L}ukasz and Krzywda, Jan A},
  journal={arXiv preprint arXiv:2510.13578},
  year={2024}
}
```
