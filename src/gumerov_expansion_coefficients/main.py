# https://github.com/search?q=gumerov+translation+language%3APython&type=code&l=Python
from typing import ParamSpec, TypeVar

from array_api._2024_12 import Array, ArrayNamespace
from array_api_compat import array_namespace
from scipy.special import sph_harm, spherical_jn, spherical_yn

P = ParamSpec("P")
T = TypeVar("T")


# (2.14)
def R(n: Array, m: Array, kr: Array, theta: Array, phi: Array) -> Array:
    """Regular elementary solution of 3D Helmholtz equation."""
    return spherical_jn(n, kr) * sph_harm(n, m, theta, phi)


def S(n: Array, m: Array, kr: Array, theta: Array, phi: Array) -> Array:
    """Singular elementary solution of 3D Helmholtz equation."""
    return spherical_yn(n, kr) * sph_harm(n, m, theta, phi)


# Gumerov's notation
# E^m_n = sum_{m'n'} (E|F)^{m' m}_{n' n} F^{m'}_{n'}
# (E|F)^{m' m}_{n'} := (E|F)^{m' m}_{n' |m|}
# (E|F)^{m' m}_{,n} := (E|F)^{m' m}_{|m'| n}


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


def minus_1_power(x: Array, /) -> Array:
    return 1 - 2 * (x % 2)


def a(n: Array, m: Array, /) -> Array:
    xp = array_namespace(n, m)
    m_abs = xp.abs(m)
    return xp.where(
        m_abs > n,
        0,
        xp.sqrt((n + m_abs + 1) * (n - m_abs + 1) / ((2 * n + 1) * (2 * n + 3))),
    )


def b(n: Array, m: Array, /) -> Array:
    xp = array_namespace(n, m)
    return xp.where(
        xp.abs(m) > n,
        0,
        xp.sqrt((n - m - 1) * (n - m) / ((2 * n - 1) * (2 * n + 1))) * xp.sign(n),
    )


def translational_coefficients(
    kr: Array, theta: Array, phi: Array, same: bool, n_end: int, /
) -> Array:
    xp = array_namespace(kr, theta, phi)
    result = xp.zeros(
        (ndim_harm(n_end), ndim_harm(n_end)), dtype=kr.dtype, device=kr.device
    )
    n, m = idx_all(n_end, xp=xp)
    # (E|F)^{m' 0}_{n' 0} = (E|F)^{m' 0}_{n'}
    if not same:
        # 4.43
        result[:, idx(0, 0)] = xp.sqrt(4 * xp.pi) * S(n, -m, -kr, theta, phi)
    else:
        # 4.58
        result[:, idx(0, 0)] = xp.sqrt(4 * xp.pi) * R(n, -m, kr, theta, phi)
    # 4.67
    # result
