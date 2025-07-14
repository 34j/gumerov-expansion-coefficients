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


def idx_i(n: int, m: int, /) -> int:
    """Index for the coefficients."""
    # (0, 0) -> 0
    # (1, -1) -> 1
    # (1, 0) -> 2
    if abs(m) > n:
        return -1
    return n * (n + 1) + m


def idx(n: Array, m: Array, /) -> Array:
    """Index for the coefficients."""
    # (0, 0) -> 0
    # (1, -1) -> 1
    # (1, 0) -> 2
    xp = array_namespace(n, m)
    m_abs = xp.abs(m)
    return xp.where(m_abs > n, -1, n * (n + 1) + m)


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
    m_abs = xp.abs(m)
    return xp.where(
        m_abs > n,
        0,
        xp.sqrt((n - m - 1) * (n - m) / ((2 * n - 1) * (2 * n + 1))) * xp.sign(n),
    )


def translational_coefficients_sectorial_init(
    kr: Array, theta: Array, phi: Array, same: bool, n_end: int, /
) -> Array:
    """Initial values of sectorial translational coefficients (E|F)^{m',m}_{0, 0}

    Parameters
    ----------
    kr : Array
        k * r of shape (...,)
    theta : Array
        polar angle of shape (...,)
    phi : Array
        azimuthal angle of shape (...,)
    same : bool
        If True, return (R|R) = (S|S).
        If False, return (S|R).
    n_end : int
        Maximum degree of spherical harmonics.

    Returns
    -------
    Array
        Initial sectorial translational coefficients of shape (ndim_harm(n_end),)
    """
    xp = array_namespace(kr, theta, phi)
    n, m = idx_all(n_end, xp=xp)
    # (E|F)^{m' 0}_{n' 0} = (E|F)^{m' 0}_{n'}
    if not same:
        # 4.43
        return minus_1_power(n) * xp.sqrt(4 * xp.pi) * S(n, -m, kr, theta, phi)
    else:
        # 4.58
        return minus_1_power(n) * xp.sqrt(4 * xp.pi) * R(n, -m, kr, theta, phi)


def translational_coefficients_sectorial(
    kr: Array,
    theta: Array,
    phi: Array,
    same: bool,
    n_end: int,
    /,
    *,
    translational_coefficients_sectorial_init: Array,
) -> Array:
    """Sectorial translational coefficients (E|F)^{m',m}_{n',|m|}

    Parameters
    ----------
    kr : Array
        k * r of shape (...,)
    theta : Array
        polar angle of shape (...,)
    phi : Array
        azimuthal angle of shape (...,)
    same : bool
        If True, return (R|R) = (S|S).
        If False, return (S|R).
    n_end : int
        Maximum degree of spherical harmonics.
    translational_coefficients_sectorial_init : Array
        Initial sectorial translational coefficients of shape (ndim_harm(n_end),)

    Returns
    -------
    Array
        Sectorial translational coefficients of shape (ndim_harm(n_end), 2*n_end-1)
    """
    xp = array_namespace(kr, theta, phi)
    result = xp.zeros(
        (ndim_harm(2 * n_end), 2 * n_end - 1), dtype=kr.dtype, device=kr.device
    )
    result[:, 0] = translational_coefficients_sectorial_init
    # 4.67
    for m in range(n_end):
        nd, md = idx_all(n_end, xp=xp)
        result[idx(nd, md), m + 1] = (
            1
            / b(m + 1, -m - 1)
            * (
                b(nd, -md) * result[idx(nd - 1, md - 1), m]
                - b(nd + 1, md - 1) * result[idx(nd + 1, md - 1), m]
            )
        )
    # 4.68
    for m in range(n_end):
        m = -m
        nd, md = idx_all(n_end, xp=xp)
        result[idx(nd, md), -m - 1] = (
            1
            / b(m + 1, -m - 1)
            * (
                b(nd, md) * result[idx(nd - 1, md + 1), -m]
                - b(nd + 1, -md - 1) * result[idx(nd + 1, md + 1), -m]
            )
        )
    return result
