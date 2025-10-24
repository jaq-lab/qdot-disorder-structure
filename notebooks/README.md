# Notebooks Directory

This directory contains Jupyter notebooks for analyzing the Statistical Structure of Charge Disorder in Si/SiGe Quantum Dots.

## Contents

### `Fig234.ipynb`
Analysis notebook for Figures 2, 3, and 4 from the paper. This notebook contains:
- Data loading and preprocessing
- Covariance matrix analysis
- Principal Component Analysis (PCA) implementation
- Visualization of disorder modes
- Statistical analysis of quantum dot parameters

### `Fig5.ipynb`
Analysis notebook for Figure 5 from the paper. This notebook focuses on:
- Control mode analysis
- Comparison of plunger-only vs three-gate control schemes
- Controllability assessment
- Gate voltage sweep analysis

### `Fig6.ipynb`
Analysis notebook for Figure 6 from the paper. This notebook includes:
- Advanced statistical analysis
- Correlation matrix visualization
- Disorder mode characterization
- Additional supporting analysis

## Usage

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

3. **Open notebooks** in the following order for complete analysis:
   - Start with `Fig234.ipynb` for basic analysis
   - Continue with `Fig5.ipynb` for control analysis
   - Finish with `Fig6.ipynb` for advanced statistics

## Data Requirements

The notebooks expect the following data files in the `data/` directory:
- `cov_matrix_raw_1e10.npy`
- `cov_matrix_raw_5e9.npy`
- `cov_matrix_pca_1e10.npy`
- `cov_matrix_pca_5e9.npy`
- `cov_matrix_artificial_1e10.npy`
- `cov_matrix_artificial_5e9.npy`

## Key Analysis Steps

1. **Data Loading**: Load covariance matrices and parameter data
2. **PCA Analysis**: Apply Principal Component Analysis to identify disorder modes
3. **Visualization**: Create plots showing disorder modes and correlations
4. **Control Analysis**: Compare different control schemes
5. **Statistical Analysis**: Calculate variance explained and controllability metrics

## Citation

If you use these notebooks in your research, please cite our paper:
```bibtex
@article{samadi2024statistical,
  title={Statistical Structure of Charge Disorder in Si/SiGe Quantum Dots},
  author={Samadi, Saeed and Cywi{\'n}ski, {\L}ukasz and Krzywda, Jan A},
  journal={arXiv preprint arXiv:2510.13578},
  year={2024}
}
```
