# https://github.com/search?q=gumerov+translation+language%3APython&type=code&l=Python
from types import EllipsisType
from typing import ParamSpec, TypeVar

from array_api._2024_12 import Array, ArrayNamespace
from array_api_compat import array_namespace
from scipy.special import sph_harm_y, spherical_jn, spherical_yn

P = ParamSpec("P")
T = TypeVar("T")


# (2.14)
def R(n: Array, m: Array, kr: Array, theta: Array, phi: Array) -> Array:
    """Regular elementary solution of 3D Helmholtz equation."""
    xp = array_namespace(n, m, kr, theta, phi)
    return xp.asarray(spherical_jn(n, kr) * sph_harm_y(n, m, theta, phi))


def S(n: Array, m: Array, kr: Array, theta: Array, phi: Array) -> Array:
    """Singular elementary solution of 3D Helmholtz equation."""
    xp = array_namespace(n, m, kr, theta, phi)
    return xp.asarray(spherical_yn(n, kr) * sph_harm_y(n, m, theta, phi))


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


def idx(n: Array | int, m: Array | int, /) -> Array:
    """Index for the coefficients."""
    # (0, 0) -> 0
    # (1, -1) -> 1
    # (1, 0) -> 2
    xp = array_namespace(n, m)
    m_abs = xp.abs(m)
    return xp.where(m_abs > n, -1, n * (n + 1) + m)


def idx_all(n_end: int, /, xp: ArrayNamespace) -> tuple[Array, Array]:
    n = xp.arange(n_end, dtype=xp.int32)[:, None]
    m = xp.arange(-n_end + 1, n_end, dtype=xp.int32)[None, :]
    n, m = xp.broadcast_arrays(n, m)
    mask = n >= xp.abs(m)
    return n[mask], m[mask]


def ndim_harm(n_end: int, /) -> int:
    """Number of spherical harmonics which degree is less than n_end."""
    return n_end**2


def minus_1_power(x: Array, /) -> Array:
    return 1 - 2 * (x % 2)


def a(n: Array | int, m: Array | int, /) -> Array:
    xp = array_namespace(n, m)
    m_abs = xp.abs(m)
    return xp.where(
        m_abs > n,
        0,
        xp.sqrt((n + m_abs + 1) * (n - m_abs + 1) / ((2 * n + 1) * (2 * n + 3))),
    )


def b(n: Array | int, m: Array | int, /) -> Array:
    xp = array_namespace(n, m)
    m_abs = xp.abs(m)
    return xp.where(
        m_abs > n,
        0,
        xp.sqrt((n - m - 1) * (n - m) / ((2 * n - 1) * (2 * n + 1))) * xp.sign(n),
    )


def getitem_outer_zero(
    array: Array,
    indices: tuple[int | slice | EllipsisType | Array | None, ...],
    /,
    *,
    axis: int = 0,
) -> Array:
    len_axis = array.shape[axis]
    index_axis = indices[axis]
    array = array[indices]
    array[(index_axis < 0) | (index_axis >= len_axis)] = 0  # type: ignore
    return array


def translational_coefficients_sectorial_init(
    kr: Array, theta: Array, phi: Array, same: bool, n_end: int, /
) -> Array:
    """Initial values of sectorial translational coefficients (E|F)^{m',0}_{n', 0}

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
    n, m = idx_all(2 * n_end - 1, xp=xp)
    # (E|F)^{m' 0}_{n' 0} = (E|F)^{m' 0}_{n'}
    if not same:
        # 4.43
        return minus_1_power(n) * xp.sqrt(xp.asarray(4.0) * xp.pi) * S(n, -m, kr, theta, phi)
    else:
        # 4.58
        return minus_1_power(n) * xp.sqrt(xp.asarray(4.0) * xp.pi) * R(n, -m, kr, theta, phi)


def translational_coefficients_sectorial(
    *,
    n_end: int,
    translational_coefficients_sectorial_init: Array,
) -> Array:
    """Sectorial translational coefficients (E|F)^{m',m}_{n',n=|m|}

    Parameters
    ----------
    n_end : int
        Maximum degree of spherical harmonics.
    translational_coefficients_sectorial_init : Array
        Initial sectorial translational coefficients of shape (ndim_harm(n_end),)

    Returns
    -------
    Array
        Sectorial translational coefficients [(m',n'),m] of shape (ndim_harm(n_end), 2*n_end-1).
        While the array shape is redundant, we give up further optimization
        because the batched axis in calculation (first) and used axis (last) are different.
    """
    xp = array_namespace(translational_coefficients_sectorial_init)
    dtype = translational_coefficients_sectorial_init.dtype
    device = translational_coefficients_sectorial_init.device
    result = xp.zeros((ndim_harm(2 * n_end - 1), 2 * n_end - 1), dtype=dtype, device=device)
    result[:, 0] = translational_coefficients_sectorial_init
    # 4.67
    for m in range(n_end - 1):
        nd, md = idx_all(n_end, xp=xp)
        result[idx(nd, md), m + 1] = (
            1
            / b(xp.asarray(m + 1), xp.asarray(-m - 1))
            * (
                b(nd, -md) * getitem_outer_zero(result, (idx(nd - 1, md - 1), m))
                - b(nd + 1, md - 1) * getitem_outer_zero(result, (idx(nd + 1, md - 1), m))
            )
        )
    # 4.68
    for m in range(n_end - 1):
        m = -m
        nd, md = idx_all(n_end, xp=xp)
        result[idx(nd, md), m - 1] = (
            1
            / b(xp.asarray(m + 1), xp.asarray(-m - 1))
            * (
                b(nd, md) * getitem_outer_zero(result, (idx(nd - 1, md + 1), m))
                - b(nd + 1, -md - 1) * getitem_outer_zero(result, (idx(nd + 1, md + 1), m))
            )
        )
    return result


def translational_coefficients_iter(
    *,
    m: int,
    md: int,
    n_end: int,
    translational_coefficients_sectorial_m_n: Array,
    translational_coefficients_sectorial_md_nd: Array,
) -> Array:
    xp = array_namespace(
        translational_coefficients_sectorial_m_n, translational_coefficients_sectorial_md_nd
    )
    dtype = translational_coefficients_sectorial_m_n.dtype
    device = translational_coefficients_sectorial_m_n.device
    mabs = abs(m)
    mdabs = abs(md)
    mlarger = max(mabs, mdabs)
    sized = 2 * n_end - mdabs - mlarger - 1
    size = 2 * n_end - mabs - mlarger - 1
    n_iter = n_end - mlarger  # [nd, n]
    md_m_fixed = xp.zeros((sized, size), dtype=dtype, device=device)
    md_m_fixed[:, 0] = translational_coefficients_sectorial_m_n[
        idx(xp.arange(mdabs, 2 * n_end - mlarger, device=device, dtype=xp.int32), md), m
    ]
    md_m_fixed[0, :] = translational_coefficients_sectorial_md_nd[
        md, idx(xp.arange(mabs, 2 * n_end - mlarger, device=device, dtype=xp.int32), m)
    ]
    # batch for nd, grow n
    mabss = (
        (mabs, mdabs),
        (mdabs, mabs),
    )
    del mabs, mdabs
    for m1abs, m2abs in mabss:
        for i in range(n_iter):
            # 4.26, 2nd term is the result
            n1 = slice(m1abs + i + 1, 2 * n_end - mlarger - i - 1)
            n1f = xp.arange(n1.start, n1.stop, dtype=dtype, device=device)
            n2 = i + m2abs
            md_m_n2_fixed = (
                -a(n1f, md) * md_m_fixed[i + 2 : -i, i]  # 3rd
                + a(n1f - 1, md) * md_m_fixed[i : -i - 2, i]  # 4th
            )
            if i > 0:
                md_m_n2_fixed += a(n2 - 1, m) * md_m_fixed[i + 1 : -i - 1, i - 1]  # 1st
            md_m_fixed[i + 1 : -i - 1, i + 1] = md_m_n2_fixed / a(n2, m)
        md_m_fixed = xp.moveaxis(md_m_fixed, 0, 1)
    return md_m_fixed[: n_end - abs(md), : n_end - abs(m)]


def translational_coefficients_all(
    *,
    n_end: int,
    translational_coefficients_sectorial_m_n: Array,
    translational_coefficients_sectorial_md_nd: Array,
) -> Array:
    """Translational coefficients (E|F)^{m',m}_{n',n'}

    Parameters
    ----------
    n_end : int
        Maximum degree of spherical harmonics.
    translational_coefficients_sectorial_m_n : Array
        Sectorial translational coefficients [(m',n'),m] of shape (ndim_harm(n_end), 2*n_end-1)
    translational_coefficients_sectorial_md_nd : Array
        Sectorial translational coefficients [m',(m,n)] of shape (2*n_end-1, ndim_harm(n_end))

    Returns
    -------
    Array
        Translational coefficients [(m',n'),(m,n)] of shape (ndim_harm(n_end), ndim_harm(n_end))
    """
    xp = array_namespace(
        translational_coefficients_sectorial_m_n, translational_coefficients_sectorial_md_nd
    )
    dtype = translational_coefficients_sectorial_m_n.dtype
    device = translational_coefficients_sectorial_m_n.device
    result = xp.zeros((ndim_harm(n_end), ndim_harm(n_end)), dtype=dtype, device=device)
    for m in range(-n_end + 1, n_end):
        for md in range(-n_end + 1, n_end):
            n = xp.arange(abs(m), n_end, dtype=xp.int32, device=device)[None, :]
            nd = xp.arange(abs(md), n_end, dtype=xp.int32, device=device)[:, None]
            result[idx(nd, md), idx(n, m)] = translational_coefficients_iter(
                m=m,
                md=md,
                n_end=n_end,
                translational_coefficients_sectorial_m_n=translational_coefficients_sectorial_m_n,
                translational_coefficients_sectorial_md_nd=translational_coefficients_sectorial_md_nd,
            )
    return result


def translational_coefficients(
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
    translational_coefficients_sectorial_init_ = translational_coefficients_sectorial_init(
        kr, theta, phi, same, n_end
    )
    translational_coefficients_sectorial_m_n = translational_coefficients_sectorial(
        n_end=n_end,
        translational_coefficients_sectorial_init=translational_coefficients_sectorial_init_,
    )
    n = xp.arange(n_end)[:, None]
    nd = idx_all(n_end, xp=xp)[0][None, :]
    # 4.61
    translational_coefficients_sectorial_md_nd = (
        minus_1_power(n + nd) * translational_coefficients_sectorial_m_n.T
    )
    return translational_coefficients_all(
        n_end=n_end,
        translational_coefficients_sectorial_m_n=translational_coefficients_sectorial_m_n,
        translational_coefficients_sectorial_md_nd=translational_coefficients_sectorial_md_nd,
    )
