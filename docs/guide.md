<!-- docs/guide.md -->

# HCR-NN Library Guide

Welcome to the **HCR-NN Library**! This guide walks you through installation, basic usage, and advanced workflows using the core API.

## 1. Overview

The HCR-NN Library implements **Hierarchical Correlation Reconstruction (HCR)** using neural approaches. Key highlights:

-  Estimates conditional expectations, densities, and probabilistic relationships.
-  Supports flexible basis functions: Legendre polynomials, cosine basis, or KDE.
-  Integrates with PyTorch modules such as `HCRCond2D` for trainable models and `HCRNeuron`.
-  Enables analysis, uncertainty estimation, and probabilistic inference in multivariate data.

## 2. Installation

Ensure your environment has:

- Python 3.9 or newer  
- [PyTorch](https://pytorch.org/)  
- NumPy  
- SciPy  
- pytest (for running tests)  

Install with:

```bash
pip install -r requirements.txt
```

Normalize data to [0,1]:

```python
norm = CDFNorm(method='empirical')
u1 = norm(torch.tensor(raw_x))
u2 = norm(torch.tensor(raw_y))
```

Build the model:

```python
model = build_hcr_cond2d(degree=8, basis='polynomial', quantile_fn=your_quantile_fn)
```

Train the model and visualize the conditional expectation curve using `model.conditional_curve().`

4. Example Workflows

End-to-end training: Use the `01_hcrnn_end_to_end.ipynb` notebook in example_implementation/ for a full demonstration — from raw data, through normalization, training, and visualization.

Density-only utilities: Calculate conditional densities directly using the density module for exploratory analysis.

Use of HCRNeuron: Demonstrates feature extraction via basis expansion using HCRNeuron.

5. Advanced Configuration

Basis Selection: Choose between `polynomial`, `cosine`, or `kde` basis using `select_basis()` or via `HCRCond2D(basis_name=...).`

Coefficient Initialization: Control initialization with `coeff_init` parameter (zeros, eye, or xavier).

Grid Resolution: Adjust grid_size for finer/purer visualizations of density curves.

Quantile Functions: Provide custom normalization via quantile_fn when working with raw data.

6. Project Structure
```
HCR-NN-Library/
├── hcr_nn/            # Core library modules (basis, layers, models, density, neuron)
├── data/              # Demo datasets or generated data
├── examples/          # Notebooks demonstrating workflows
├── docs/              # This guide and future documentation
├── tests_python/      # Unit and integration tests (e.g. nb execution)
├── README.md          # Project landing page (below)
├── requirements.txt   # Pinned environment and dependencies
```
7. Contributing

We welcome contributions! Please:

Fork the repository and open a pull request with clear scope.

Write clear documentation, docstrings, and tests for new features.

Ensure notebooks in examples/ run successfully in CI (pytest).

Follow Python and project style conventions (PEP 8, docstring formats, type hints).

8. Further Reading

HCR Theory: Refer to the original paper (arXiv:2405.05097) for theoretical foundations (normalization, orthonormal basis, propagation).

Code Documentation: Read inline docstrings in hcr_nn/basis.py, models.py, density.py, layers.py, neuron.py for API-specific details.

That’s it—let this guide be your starting point for exploring and building with HCR‑NN!