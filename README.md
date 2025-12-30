# HCR-NN Library

![Python version](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A professional, modular Python library for **Hierarchical Correlation Reconstruction with Neural Networks (HCR-NN)**.

Recent PyPI release: https://pypi.org/project/hcr-nn/

Official GitHub repository: https://github.com/mateuszwalo/HCR-NN-Library

---

##  What Is HCR-NN?

HCR-NN enables efficient density estimation and conditional inference on multivariate data using principled statistical modeling wrapped in modern neural network constructs. It is designed for both research and application:

- Model **joint and conditional densities**, e.g., \(ρ(u_1, u_2)\) and \(E[u_1\mid u_2]\).  
- Normalize raw data using empirical or Gaussian CDFs (`CDFNorm`).  
- Approximate densities via **basis expansions** (polynomial, cosine, KDE).  
- Build, train, and visualize **conditional models** with `HCRCond2D`.  
- Integrate with aurhus-propositional modeling workflows & PyTorch pipelines.

---

## ​ Highlights

- **Precise probabilistic modeling** with neural network flexibility.  
- **Modular design**: swap between normalization, basis, model easily.  
- **Reproducible example notebooks** (`examples/`) that run in CI and output artifacts.  
- **Optimized density utilities** for direct density inference and conditional expectation.

---

<!-- ##  Quick Start

```bash
git clone https://github.com/mateuszwalo/HCR-NN-Library.git
cd HCR-NN-Library
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt  # includes torch, numpy, pandas, pytest, etc.
``` -->

## Example Workflow: Train and Visualize

```bash
jupyter notebook example_implementation/01_hcrnn_end_to_end.ipynb
```

This notebook demonstrates loading data, normalization, training HCRCond2D, and visualizing 𝐸[𝑢1∣𝑢2]E[u1∣u2].

## Updates

### 30 XII 25'

Optimized core code functions by implementing by implementing mixed Just-in-Time Compilation for small scale computations and CUDA kernels for big scale calculations. The implementations are prototyped in dedicated CU format and implemented into code using Python Numba interface. Why specifically Numba and not C? We decided to follow Numba approach to facilitate the process of integrating GPU-based code and experimenting of science team along with existing memory management solution. Full mathematical implementation with explanation and benchmarking is shown in the [short pdf paper](/HCR-NN-Library/docs/Optimizing%20HCR%20functions.pdf).

## Documentation

Getting started & API reference: see docs/guide.md.

In-code documentation: each module in hcr_nn/ is documented via docstrings.

Further reading: the foundational theory is detailed in [the HCR‑NN paper (arXiv:2405.05097)].

## License

This project is available under the MIT License. See LICENSE for details.

## Acknowledgments

Developed by contributors from AIntern:

****Mateusz Walo****

****Felicja Warno****

****Adrian Przybysz****

****Karol Rochalski****

****Zuzanna Sieńko****

****Under the supervision of PhD Jarek Duda author HCR-NN theory****

Inspired by HCR Theory: Refer to the original paper (arXiv:2405.05097) for theoretical foundations (normalization, orthonormal basis, propagation).


## Contributing

We welcome contributions of any kind:

Fork the repo and create a branch for your feature or fix.

Add clear documentation and tests (preferably notebooks or pytest) — all examples should run in CI.

Open a Pull Request with a clear description of your changes.

Check the test suite `pytest -q` in CI for regression safety.
