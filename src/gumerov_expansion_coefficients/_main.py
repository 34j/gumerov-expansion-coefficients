# https://github.com/search?q=gumerov+translation+language%3APython&type=code&l=Python
from types import EllipsisType

import numpy as np
from array_api._2024_12 import Array
from array_api_compat import array_namespace
from array_api_negative_index import flip_symmetric
from numba import prange

from gumerov_expansion_coefficients._elementary_solutions import R_all, S_all, idx_all

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


def ndim_harm(n_end: int, /) -> int:
    """Number of spherical harmonics which degree is less than n_end."""
    return n_end**2


def minus_1_power(x: Array, /) -> Array:
    return 1 - 2 * (x % 2)


def a(n: int, m: int, /) -> float:
    m_abs = abs(m)
    if m_abs > n:
        return 0
    return np.sqrt((n + m_abs + 1) * (n - m_abs + 1) / ((2 * n + 1) * (2 * n + 3)))


def b(n: int, m: int, /) -> float:
    m_abs = abs(m)
    if m_abs > n:
        return 0
    tmp = np.sqrt((n - m - 1) * (n - m) / ((2 * n - 1) * (2 * n + 1)))
    if m >= 0:
        return tmp
    else:
        return -tmp


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
    array[..., (index_axis < 0) | (index_axis >= len_axis)] = 0  # type: ignore
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
        Initial sectorial translational coefficients of shape (..., ndim_harm(n_end),)
    """
    xp = array_namespace(kr, theta, phi)
    n, m = idx_all(2 * n_end - 1, xp=xp, dtype=xp.int32, device=kr.device)
    # (E|F)^{m' 0}_{n' 0} = (E|F)^{m' 0}_{n'}
    if not same:
        # 4.43
        return (
            minus_1_power(n)
            * xp.sqrt(xp.asarray(4.0, dtype=kr.dtype, device=kr.device) * xp.pi)
            * S_all(kr, theta, phi, n_end=2 * n_end - 1)[..., idx(n, -m)]
        )
    else:
        # 4.58
        return (
            minus_1_power(n)
            * xp.sqrt(xp.asarray(4.0, dtype=kr.dtype, device=kr.device) * xp.pi)
            * R_all(kr, theta, phi, n_end=2 * n_end - 1)[..., idx(n, -m)]
        )


def translational_coefficients_sectorial_n_m(
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
        Initial sectorial translational coefficients of shape (..., ndim_harm(n_end),)

    Returns
    -------
    Array
        Sectorial translational coefficients [(m',n'),m]
        of shape (..., ndim_harm(n_end), 2*n_end-1).
        While the array shape is redundant, we give up further optimization
        because the batched axis in calculation (first) and used axis (last) are different.
    """
    xp = array_namespace(translational_coefficients_sectorial_init)
    shape = translational_coefficients_sectorial_init.shape[:-1]
    dtype = translational_coefficients_sectorial_init.dtype
    device = translational_coefficients_sectorial_init.device
    result = xp.zeros((*shape, ndim_harm(2 * n_end - 1), 4 * n_end - 1), dtype=dtype, device=device)
    result[..., :, 0] = translational_coefficients_sectorial_init
    if dtype == xp.complex64:
        dtype = xp.float32
    elif dtype == xp.complex128:
        dtype = xp.float64
    # 4.67
    for m in range(2 * n_end - 2):
        nd, md = idx_all(2 * n_end - abs(m) - 2, xp=xp, dtype=xp.int32, device=device)
        result[..., idx(nd, md), m + 1] = (
            1
            / b(
                xp.asarray(m + 1, dtype=dtype, device=device),
                xp.asarray(-m - 1, dtype=dtype, device=device),
            )
            * (
                b(nd, -md) * getitem_outer_zero(result, (..., idx(nd - 1, md - 1), m), axis=-2)
                - b(nd + 1, md - 1)
                * getitem_outer_zero(result, (..., idx(nd + 1, md - 1), m), axis=-2)
            )
        )
    # 4.68
    for m in range(2 * n_end - 2):
        nd, md = idx_all(2 * n_end - abs(m) - 2, xp=xp, dtype=xp.int32, device=device)
        result[..., idx(nd, md), -m - 1] = (
            1
            / b(
                xp.asarray(m + 1, dtype=dtype, device=device),
                xp.asarray(-m - 1, dtype=dtype, device=device),
            )
            * (
                b(nd, md) * getitem_outer_zero(result, (..., idx(nd - 1, md + 1), -m), axis=-2)
                - b(nd + 1, -md - 1)
                * getitem_outer_zero(result, (..., idx(nd + 1, md + 1), -m), axis=-2)
            )
        )
    return result


def translational_coefficients_sectorial_nd_md(
    *,
    n_end: int,
    translational_coefficients_sectorial_n_m: Array,
) -> Array:
    """Sectorial translational coefficients (E|F)^{m',m}_{n'=|m'|,n}

    Parameters
    ----------
    n_end : int
        Maximum degree of spherical harmonics.
    translational_coefficients_sectorial_n_m : Array
        Initial sectorial translational coefficients of shape (..., ndim_harm(n_end), 2*n_end-1)

    Returns
    -------
    Array
        Sectorial translational coefficients [m',(m,n)] of shape (..., 2*n_end-1, ndim_harm(n_end)).
        While the array shape is redundant, we give up further optimization
        because the batched axis in calculation (first) and used axis (last) are different.
    """
    xp = array_namespace(translational_coefficients_sectorial_n_m)
    device = translational_coefficients_sectorial_n_m.device
    m = xp.concat(
        (
            xp.arange(2 * n_end, dtype=xp.int32, device=device),
            xp.arange(-2 * n_end + 1, 0, dtype=xp.int32, device=device),
        ),
        axis=0,
    )
    n = xp.abs(m)
    nd, md = idx_all(2 * n_end - 1, xp=xp, dtype=xp.int32, device=device)
    # 4.61
    return (
        minus_1_power(n[:, None] + nd[None, :])
        * flip_symmetric(xp.moveaxis(translational_coefficients_sectorial_n_m, -1, -2), axis=-2)[
            ..., idx(nd, -md)
        ]
    )


def _translational_coefficients_all(
    *,
    n_end: int,
    translational_coefficients_sectorial_init: Array,
    result: Array,
) -> None:
    """Translational coefficients (E|F)^{m',m}_{n',n}

    Parameters
    ----------
    n_end : int
        Maximum degree of spherical harmonics.
    translational_coefficients_sectorial_init : Array
        Initial sectorial translational coefficients (E|F)^{m',0}_{n', 0}
        of shape (..., ndim_harm(n_end),)
    result : Array
        Translational coefficients [(m',n'),(m,n)] of shape (ndim_harm(n_end), ndim_harm(n_end))
    """

    for m in prange(-n_end + 1, n_end):
        for md in prange(-n_end + 1, n_end):
            mabs, mdabs = abs(m), abs(md)
            mlarger = max(mabs, mdabs)
            n_iter = n_end - mlarger - 1

            # init
            for n in prange(mabs, 2 * n_end - mlarger + 1):
                result[idx_i(abs(md), md), idx_i(n, m)] = (
                    translational_coefficients_sectorial_md_nd[md, idx_i(n, m)]
                )
            for nd in prange(mdabs, 2 * n_end - mlarger + 1):
                result[idx_i(nd, md), idx_i(abs(m), m)] = translational_coefficients_sectorial_m_n[
                    idx_i(nd, md), m
                ]

            for m1_is_md, m1, m2 in ((True, md, m), (False, m, md)):
                del m, md
                m1abs = abs(m1)
                m2abs = abs(m2)
                for i in range(n_iter):
                    for n1 in prange(m1abs + i + 1, 2 * n_end - mlarger - i - 2):
                        n2 = i + m2abs
                        tmp = (
                            -a(n1, m1)
                            * result[
                                (idx_i(n1 + 1, m1), n2) if m1_is_md else (n2, idx_i(n1 + 1, m1))
                            ]  # 3rd
                            + a(n1 - 1, m1)
                            * result[
                                (idx_i(n1 - 1, m1), n2) if m1_is_md else (n2, idx_i(n1 - 1, m1))
                            ]  # 4th
                        )
                        if i > 0:
                            tmp += (
                                a(n2 - 1, m2)
                                * result[
                                    (idx_i(n1, m1), n2 - 1) if m1_is_md else (n2 - 1, idx_i(n1, m1))
                                ]
                            )  # 1st
                        result[(idx_i(n1, m1), n2 + 1) if m1_is_md else (n2 + 1, idx_i(n1, m1))] = (
                            tmp / a(n2, m2)
                        )
    return result


def translational_coefficients_all(
    *,
    n_end: int,
    translational_coefficients_sectorial_m_n: Array,
    translational_coefficients_sectorial_md_nd: Array,
) -> Array:
    """Translational coefficients (E|F)^{m',m}_{n',n}

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
    shape = translational_coefficients_sectorial_m_n.shape[:-2]
    result = xp.zeros((*shape, ndim_harm(n_end), ndim_harm(n_end)), dtype=dtype, device=device)
    return xp.asarray(
        _translational_coefficients_all(
            n_end=n_end,
            translational_coefficients_sectorial_m_n=translational_coefficients_sectorial_m_n,
            translational_coefficients_sectorial_md_nd=translational_coefficients_sectorial_md_nd,
            result=result,
        ),
        dtype=dtype,
        device=device,
    )


def translational_coefficients(
    kr: Array, theta: Array, phi: Array, *, same: bool, n_end: int
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
    translational_coefficients_sectorial_init_ = translational_coefficients_sectorial_init(
        kr, theta, phi, same, n_end
    )
    translational_coefficients_sectorial_n_m_ = translational_coefficients_sectorial_n_m(
        n_end=n_end,
        translational_coefficients_sectorial_init=translational_coefficients_sectorial_init_,
    )
    translational_coefficients_sectorial_nd_md_ = translational_coefficients_sectorial_nd_md(
        n_end=n_end,
        translational_coefficients_sectorial_n_m=translational_coefficients_sectorial_n_m_,
    )
    return translational_coefficients_all(
        n_end=n_end,
        translational_coefficients_sectorial_m_n=translational_coefficients_sectorial_n_m_,
        translational_coefficients_sectorial_md_nd=translational_coefficients_sectorial_nd_md_,
    )
