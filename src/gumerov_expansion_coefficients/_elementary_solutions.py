# These functions are not JIT-compatible, thus we use array_api_compat.
# (2.14)
from typing import Any

from array_api._2024_12 import Array, ArrayNamespace
from array_api_compat import array_namespace, to_device
from array_api_compat import numpy as np
from scipy.special import sph_harm_y_all, spherical_jn, spherical_yn


def idx_all(n_end: int, /, xp: ArrayNamespace, dtype: Any, device: Any) -> tuple[Array, Array]:
    dtype = dtype or xp.int32
    n = xp.arange(n_end, dtype=dtype, device=device)[:, None]
    m = xp.arange(-n_end + 1, n_end, dtype=dtype, device=device)[None, :]
    n, m = xp.broadcast_arrays(n, m)
    mask = n >= xp.abs(m)
    return n[mask], m[mask]


def R_all(kr: Array, theta: Array, phi: Array, *, n_end: int) -> Array:
    """Regular elementary solution of 3D Helmholtz equation.

    Parameters
    ----------
    kr : Array
        k * r of shape (...,)
    theta : Array
        polar angle of shape (...,)
    phi : Array
        azimuthal angle of shape (...,)
    n_end : int
        Maximum degree of spherical harmonics.

    Returns
    -------
    Array
        Regular elementary solution of 3D Helmholtz equation of shape (..., ndim_harm(n_end),)
    """
    xp = array_namespace(kr, theta, phi)
    device = kr.device
    dtype = kr.dtype
    if dtype == xp.float32:
        dtype = xp.complex64
    elif dtype == xp.float64:
        dtype = xp.complex128
    n, m = idx_all(n_end, xp=xp, dtype=xp.int32, device="cpu")
    kr = to_device(kr, "cpu")
    theta = to_device(theta, "cpu")
    phi = to_device(phi, "cpu")
    return xp.asarray(
        spherical_jn(n, kr[..., None])
        * np.moveaxis(sph_harm_y_all(n_end - 1, n_end - 1, theta, phi)[n, m, ...], 0, -1),
        dtype=dtype,
        device=device,
    )


def S_all(kr: Array, theta: Array, phi: Array, *, n_end: int) -> Array:
    """Singular elementary solution of 3D Helmholtz equation.

    Parameters
    ----------
    kr : Array
        k * r of shape (...,)
    theta : Array
        polar angle of shape (...,)
    phi : Array
        azimuthal angle of shape (...,)
    n_end : int
        Maximum degree of spherical harmonics.

    Returns
    -------
    Array
        Singular elementary solution of 3D Helmholtz equation of shape (..., ndim_harm(n_end),)"""
    xp = array_namespace(kr, theta, phi)
    device = kr.device
    dtype = kr.dtype
    if dtype == xp.float32:
        dtype = xp.complex64
    elif dtype == xp.float64:
        dtype = xp.complex128
    n, m = idx_all(n_end, xp=xp, dtype=xp.int32, device="cpu")
    kr = to_device(kr, "cpu")
    theta = to_device(theta, "cpu")
    phi = to_device(phi, "cpu")
    return xp.asarray(
        (spherical_jn(n, kr[..., None]) + 1j * spherical_yn(n, kr[..., None]))
        * np.moveaxis(sph_harm_y_all(n_end - 1, n_end - 1, theta, phi)[n, m, ...], 0, -1),
        dtype=dtype,
        device=device,
    )
