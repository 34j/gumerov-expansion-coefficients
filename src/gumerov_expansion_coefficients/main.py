# https://github.com/search?q=gumerov+translation+language%3APython&type=code&l=Python
from typing import ParamSpec, TypeVar

from array_api._2024_12 import Array, ArrayNamespace
from array_api_compat import array_namespace
from scipy.special import sph_harm, spherical_jn, spherical_yn

P = ParamSpec("P")
T = TypeVar("T")


def R(n: Array, m: Array, theta: Array, phi: Array, k: Array) -> Array:
    """Regular elementary solution of 3D Helmholtz equation."""
    return spherical_jn(n, k) * sph_harm(n, m, theta, phi)


def S(n: Array, m: Array, theta: Array, phi: Array, k: Array) -> Array:
    """Singular elementary solution of 3D Helmholtz equation."""
    return spherical_yn(n, k) * sph_harm(n, m, theta, phi)


# Gumerov's notation
# E^m_n = sum_{m'n'} (E|F)^{m' m}_{n' n} F^{m'}_{n'}


def idx(n: Array | int, m: Array | int, /) -> Array:
    """Index for the coefficients."""
    # (0, 0) -> 0
    # (1, -1) -> 1
    # (1, 0) -> 2
    return n * (n + 1) + m


def idx_all(n_end: int, /, xp: ArrayNamespace) -> tuple[Array, Array]:
    n = xp.arange(n_end, dtype=xp.int32)
    m = xp.arange(-n_end + 1, n_end, dtype=xp.int32)
    n, m = xp.meshgrid(n, m, indexing="ij")
    mask = n >= xp.abs(m)
    return n[mask], m[mask]


def ndim_harm(n_end: int, /) -> int:
    """Number of spherical harmonics which degree is less than n_end."""
    return n_end**2


def translational_coefficients(k: Array, dr: Array, same: bool, n_end: int, /) -> Array:
    xp = array_namespace(k, dr)
    result = xp.zeros(
        (ndim_harm(n_end), ndim_harm(n_end)), dtype=k.dtype, device=k.device
    )
    # (E|F)^{m' 0}_{n' 0}
    if not same:
        # 4.43
        result[:, idx(0, 0)] = xp.sqrt(4 * xp.pi)
