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
- `d` - Interdot distance
- `tcs` - Tunnel coupling strength
- `Lxavg` - Average dot size in x-direction
- `dLx` - Variation in dot size
- `Favg` - Average electric field in z-direction
- `dF` - Electric field variation in z-direction
- `eps` - Energy detuning
- `EL_orb` - Left orbital energy
- `ER_orb` - Right orbital energy
- `V_acs` - Detuning in plunger gate voltage
- `Bhs` - Barrier height
- `ratioL` - Left dot elipticity (ratio of x to y axis)
- `ratioR` - Right dot elipticity (ratio of x to y axis)

### Usage

#### Basic Data Loading
These matrices can be loaded in Python using:
```python
import numpy as np
cov_matrix = np.load('cov_matrix_raw_1e10.npy', allow_pickle=True)
```

#### Digital Twin Generation
To generate artificial quantum dot devices (digital twins) using the learned covariance structure:

```python
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

def generate_digital_twin(cov_matrix_file, n_samples=1000, seed=42):
    """
    Generate artificial quantum dot devices using multivariate Gaussian distribution.
    
    Parameters:
    -----------
    cov_matrix_file : str
        Path to covariance matrix file (e.g., 'cov_matrix_raw_1e10.npy')
    n_samples : int
        Number of artificial devices to generate
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    artificial_data : pandas.DataFrame
        Generated quantum dot parameters for n_samples devices
    """
    # Load covariance matrix and mean values
    cov_data = np.load(cov_matrix_file, allow_pickle=True)
    
    # Extract covariance matrix and mean values
    if isinstance(cov_data, pd.DataFrame):
        cov_matrix = cov_data.cov().values
        mean_values = cov_data.mean().values
        parameter_names = cov_data.columns.tolist()
    else:
        # If raw numpy array, assume it contains both mean and covariance
        cov_matrix = cov_data
        mean_values = np.zeros(cov_matrix.shape[0])  # Assume zero mean
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Generate samples from multivariate normal distribution
    artificial_samples = multivariate_normal.rvs(
        mean=mean_values, 
        cov=cov_matrix, 
        size=n_samples
    )
    
    # Convert to DataFrame
    artificial_data = pd.DataFrame(artificial_samples, columns=parameter_names)
    
    return artificial_data

# Example usage:
# Generate 1000 artificial devices using the 1e10 charge density model
artificial_devices = generate_digital_twin('cov_matrix_raw_1e10.npy', n_samples=1000)

# Generate devices using PCA-reduced model
pca_devices = generate_digital_twin('cov_matrix_pca_1e10.npy', n_samples=1000)

# Compare artificial vs real data
print("Artificial device parameters:")
print(artificial_devices.describe())
```

#### Advanced Digital Twin with Custom Parameters
```python
def generate_custom_digital_twin(cov_matrix_file, target_params=None, n_samples=1000):
    """
    Generate digital twins with specific parameter constraints.
    
    Parameters:
    -----------
    cov_matrix_file : str
        Path to covariance matrix file
    target_params : dict
        Dictionary of parameter names and target values
    n_samples : int
        Number of devices to generate
    """
    # Load base covariance structure
    cov_data = np.load(cov_matrix_file, allow_pickle=True)
    cov_matrix = cov_data.cov().values if isinstance(cov_data, pd.DataFrame) else cov_data
    mean_values = cov_data.mean().values if isinstance(cov_data, pd.DataFrame) else np.zeros(cov_matrix.shape[0])
    
    # Generate base samples
    samples = multivariate_normal.rvs(mean=mean_values, cov=cov_matrix, size=n_samples)
    
    # Apply constraints if specified
    if target_params:
        for param, target_value in target_params.items():
            if param in cov_data.columns:
                param_idx = cov_data.columns.get_loc(param)
                samples[:, param_idx] = target_value
    
    return pd.DataFrame(samples, columns=cov_data.columns)

# Example: Generate devices with specific tunnel coupling
constrained_devices = generate_custom_digital_twin(
    'cov_matrix_raw_1e10.npy',
    target_params={'tcs': 20.0},  # Target tunnel coupling of 20 μeV
    n_samples=500
)
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
