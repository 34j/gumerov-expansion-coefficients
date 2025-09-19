# Gumerov Expansion Coefficients

<p align="center">
  <a href="https://github.com/34j/gumerov-expansion-coefficients/actions/workflows/ci.yml?query=branch%3Amain">
    <img src="https://img.shields.io/github/actions/workflow/status/34j/gumerov-expansion-coefficients/ci.yml?branch=main&label=CI&logo=github&style=flat-square" alt="CI Status" >
  </a>
  <a href="https://gumerov-expansion-coefficients.readthedocs.io">
    <img src="https://img.shields.io/readthedocs/gumerov-expansion-coefficients.svg?logo=read-the-docs&logoColor=fff&style=flat-square" alt="Documentation Status">
  </a>
  <a href="https://codecov.io/gh/34j/gumerov-expansion-coefficients">
    <img src="https://img.shields.io/codecov/c/github/34j/gumerov-expansion-coefficients.svg?logo=codecov&logoColor=fff&style=flat-square" alt="Test coverage percentage">
  </a>
</p>
<p align="center">
  <a href="https://github.com/astral-sh/uv">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json" alt="uv">
  </a>
  <a href="https://github.com/astral-sh/ruff">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff">
  </a>
  <a href="https://github.com/pre-commit/pre-commit">
    <img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white&style=flat-square" alt="pre-commit">
  </a>
</p>
<p align="center">
  <a href="https://pypi.org/project/gumerov-expansion-coefficients/">
    <img src="https://img.shields.io/pypi/v/gumerov-expansion-coefficients.svg?logo=python&logoColor=fff&style=flat-square" alt="PyPI Version">
  </a>
  <img src="https://img.shields.io/pypi/pyversions/gumerov-expansion-coefficients.svg?style=flat-square&logo=python&amp;logoColor=fff" alt="Supported Python versions">
  <img src="https://img.shields.io/pypi/l/gumerov-expansion-coefficients.svg?style=flat-square" alt="License">
</p>

---

**Documentation**: <a href="https://gumerov-expansion-coefficients.readthedocs.io" target="_blank">https://gumerov-expansion-coefficients.readthedocs.io </a>

**Source Code**: <a href="https://github.com/34j/gumerov-expansion-coefficients" target="_blank">https://github.com/34j/gumerov-expansion-coefficients </a>

---

Multiple translation and rotation coefficients for the 3D Helmholtz Equation

## Installation

Install this via pip (or your favourite package manager):

```shell
pip install gumerov-expansion-coefficients
```

## Usage

```python
from gumerov_expansion_coefficients import translational_coefficients

translational_coefficients(
    k * r, theta, phi, same=True, n_end=10
)  # (R|R) coefficients from 0 to 9 th degree
translational_coefficients(
    k * r, theta, phi, same=False, n_end=10
)  # (S|R) coefficients from 0 to 9 th degree
```

- The definition of spherical harmonics are same as in [1]. Note that there are 3 other common definitions, and this definition differs from `scipy.special.sph_harm_y` for negative `m`.

$$
Y_n^m (\theta, \phi) :=
(-1)^m \sqrt{\frac{(2n+1)(n-\left|m\right|)!}{4 \pi (n+\left|m\right|)!}}
P_n^{\left|m\right|} (\cos \theta) e^{i m \phi}
$$

$$
R_n^m (kr, \theta, \phi) := j_n(kr) Y_n^m (\theta, \phi)
$$

$$
S_n^m (kr, \theta, \phi) := h_n^{(1)}(kr) Y_n^m (\theta, \phi)
$$

- The return array is 2D array with shape `(n_end**2, n_end**2)`.
- The first axis is to be summed over, resulting in the elemenary solutions at the second axis.
- The coefficient coressponding to the quantum numbers `(n, m)` is mapped to `n**2 + (m % (2 * n + 1))`-th index, while in [2] it is mapped to `n * (n + 1) + m`-th index.

## References

- [1] Gumerov, N. A., & Duraiswami, R. (2004). Recursions for the Computation of Multipole Translation and Rotation Coefficients for the 3-D Helmholtz Equation. SIAM Journal on Scientific Computing, 25(4), 1344–1381. https://doi.org/10.1137/S1064827501399705
- [2] Gumerov, N. A., & Ramani, D. (2002年). Computation of scattering from N spheres using multipole reexpansion. The Journal of the Acoustical Society of America, 112(6), 2688–2701. https://doi.org/10.1121/1.1517253

## Benchmark

```shell
gec benchmark
gec plot
```

![timing_results](https://raw.githubusercontent.com/34j/gumerov-expansion-coefficients/main/timing_results.jpg)

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- prettier-ignore-start -->
<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- markdownlint-disable -->
<!-- markdownlint-enable -->
<!-- ALL-CONTRIBUTORS-LIST:END -->
<!-- prettier-ignore-end -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!

## Credits

[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-orange.json)](https://github.com/copier-org/copier)

This package was created with
[Copier](https://copier.readthedocs.io/) and the
[browniebroke/pypackage-template](https://github.com/browniebroke/pypackage-template)
project template.
