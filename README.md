# Statistical Structure of Charge Disorder in Si/SiGe Quantum Dots

[![arXiv](https://img.shields.io/badge/arXiv-2510.13578-b31b1b.svg)](https://arxiv.org/abs/2510.13578)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org)

> **Research Repository**: This repository contains simulation data, analysis scripts, and computational notebooks for studying device variability in Si/SiGe quantum dots using Principal Component Analysis (PCA) to identify principal disorder modes and assess system controllability.

## 📄 Paper Information

**Title**: Statistical Structure of Charge Disorder in Si/SiGe Quantum Dots  
**Authors**: Saeed Samadi, Łukasz Cywiński, Jan A. Krzywda  
**arXiv**: [2510.13578](https://arxiv.org/abs/2510.13578)  
**Blog Post**: [Statistical Structure of Charge Disorder in Si/SiGe Quantum Dots](https://jan-a-krzywda.com/our-paper-statistical-structure-of-charge-disorder-in-si-sige-quantum-dots/)

## 🎯 Abstract

Properties of quantum dot based spin qubits have significant inter-device variability due to unavoidable presence of various types of disorder in semiconductor nanostructures. A significant source of this variability is charge disorder at the semiconductor-oxide interface, which causes unpredictable, yet correlated fluctuations in essential properties of quantum dots like their mutual tunnel couplings and electronic confinement energies.

This study presents a systematic approach to characterize and mitigate the effects of such disorder. We utilize finite element modeling of a Si/SiGe double quantum dot to generate a large statistical ensemble of devices, simulating the impact of trapped interface charges. This work results in a predictive statistical model capable of generating realistic artificial data for training machine learning algorithms.

## 🔬 Key Findings

- **Over 80% of device-to-device variability** is concentrated along just three dominant "disorder modes":
  - **Symmetric "squeeze or stretch"** of the two dots (biggest problem)
  - **Asymmetric "tilt"** in their energy levels  
  - **Common "vertical shift"** of the potential

- **Control Limitations**: Simplified "plunger-only" control scheme can only compensate for ~50% of disorder
- **Solution**: Three-gate control scheme can correct over 90% of disorder variance

## 📁 Repository Structure

```
qdot-disorder-structure/
├── README.md                 # This file
├── LICENSE                   # MIT License
├── requirements.txt          # Python dependencies
├── CITATION.cff             # Citation metadata
├── notebooks/               # Jupyter notebooks for analysis
│   ├── Fig234.ipynb        # Figures 2, 3, 4 analysis
│   ├── Fig5.ipynb          # Figure 5 analysis  
│   └── Fig6.ipynb          # Figure 6 analysis
├── scripts/                 # Python analysis scripts
│   └── Definitions.py      # Core analysis functions
├── data/                    # Simulation data and covariance matrices
│   ├── cov_matrix_artificial_1e10.npy
│   ├── cov_matrix_artificial_5e9.npy
│   ├── cov_matrix_pca_1e10.npy
│   ├── cov_matrix_pca_5e9.npy
│   ├── cov_matrix_raw_1e10.npy
│   └── cov_matrix_raw_5e9.npy
└── figures/                 # Generated figures and images
    └── cartoon.png         # Quantum dot visualization
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook
- Required Python packages (see `requirements.txt`)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/jaq-lab/qdot-disorder-structure.git
   cd qdot-disorder-structure
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

4. **Navigate to notebooks**:
   - Open `notebooks/Fig234.ipynb` for Figures 2, 3, 4 analysis
   - Open `notebooks/Fig5.ipynb` for Figure 5 analysis
   - Open `notebooks/Fig6.ipynb` for Figure 6 analysis

## 📊 Data Description

The repository contains three types of covariance matrices:

- **Raw Data** (`cov_matrix_raw_*.npy`): Original simulation results
- **PCA Analysis** (`cov_matrix_pca_*.npy`): Principal component analysis results  
- **Artificial Data** (`cov_matrix_artificial_*.npy`): Generated using multivariate normal distribution

Two different charge densities are studied:
- `1e10`: 1×10¹⁰ cm⁻² charge density
- `5e9`: 5×10⁹ cm⁻² charge density

## 🔧 Usage

### Core Analysis Functions

The `scripts/Definitions.py` file contains essential functions for:

- **Data Loading**: `load_data()`, `get_data_sweep()`, `get_data_map()`
- **Visualization**: `plot_sweep()`, `plot_map()`, `draw_gates()`
- **Parameter Extraction**: `get_Eav_tc_eps0()`, `calculate_lever_arm()`
- **Statistical Analysis**: `get_DQDparams()`, `calculate_error()`

### Example Usage

```python
from scripts.Definitions import load_data, plot_sweep, get_Eav_tc_eps0
import numpy as np

# Load simulation data
data = load_data('path/to/data.csv')

# Extract detuning and energy data
detuning = data[:, 0]
energy = data[:, 1:]

# Calculate quantum dot parameters
E_avg, tc, eps0 = get_Eav_tc_eps0(detuning, energy)

# Plot results
plot_sweep(plt.gca(), detuning, energy[:, 0], "Energy vs Detuning")
```

## 🎓 Citation

If you use this code or data in your research, please cite our paper:

```bibtex
@article{samadi2024statistical,
  title={Statistical Structure of Charge Disorder in Si/SiGe Quantum Dots},
  author={Samadi, Saeed and Cywi{\'n}ski, {\L}ukasz and Krzywda, Jan A},
  journal={arXiv preprint arXiv:2510.13578},
  year={2024}
}
```

## 👥 Authors

- **Saeed Samadi** - Institute of Physics, Polish Academy of Sciences
- **Łukasz Cywiński** - Institute of Physics, Polish Academy of Sciences  
- **Jan A. Krzywda** - Leiden University, Applied Quantum Algorithms

## 📧 Contact

For questions about this research or repository, please contact:
- **Jan A. Krzywda**: j.a.krzywda@liacs.leidenuniv.nl
- **Website**: [jan-a-krzywda.com](https://jan-a-krzywda.com)

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This work was supported by research funding from the Polish Academy of Sciences and Leiden University. We thank the quantum computing community for valuable discussions and feedback.

## 🔗 Related Links

- [arXiv Paper](https://arxiv.org/abs/2510.13578)
- [Author Blog Post](https://jan-a-krzywda.com/our-paper-statistical-structure-of-charge-disorder-in-si-sige-quantum-dots/)
- [GitHub Repository](https://github.com/jaq-lab/qdot-disorder-structure)

---

*This repository accompanies the research paper "Statistical Structure of Charge Disorder in Si/SiGe Quantum Dots" and provides all necessary code and data for reproducing the results.*