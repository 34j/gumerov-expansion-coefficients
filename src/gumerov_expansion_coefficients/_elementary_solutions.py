# These functions are not JIT-compatible, thus we use array_api_compat.
# (2.14)
from typing import Any, Literal, overload

from array_api._2024_12 import Array, ArrayNamespace
from array_api_compat import array_namespace, to_device
from array_api_compat import numpy as np
from array_api_negative_index import arange_asymmetric
from scipy.special import sph_harm_y_all, spherical_jn, spherical_yn


def idx_all(n_end: int, /, xp: ArrayNamespace, dtype: Any, device: Any) -> tuple[Array, Array]:
    """Get all quantum numbers (n, m) where n < n_end.

    Parameters
    ----------
    n_end : int
        Maximum degree of spherical harmonics.
    xp : ArrayNamespace
        The array namespace.
    dtype : Any
        The data type of the output arrays.
    device : Any
        The device of the output arrays.

    Returns
    -------
    tuple[Array, Array]
        Arrays of quantum numbers n and m of shape (n_end**2,).

    Examples
    --------
    >>> import array_api_compat.numpy as np
    >>> from gumerov_expansion_coefficients import idx_all
    >>> n, m = idx_all(3, xp=np, dtype=np.int32, device=None)
    >>> n
    array([0, 1, 1, 1, 2, 2, 2, 2, 2], dtype=int32)
    >>> m
    array([ 0,  0,  1, -1,  0,  1,  2, -2, -1], dtype=int32)
    """
    dtype = dtype or xp.int32
    n = xp.arange(n_end, dtype=dtype, device=device)[:, None]
    m = arange_asymmetric(n_end, xp=xp, dtype=dtype, device=device)[None, :]
    n, m = xp.broadcast_arrays(n, m)
    mask = n >= xp.abs(m)
    return n[mask], m[mask]


@overload
def minus_1_power(x: Array, /) -> Array: ...
@overload
def minus_1_power(x: int, /) -> int: ...  # type: ignore[overload-cannot-match]


def minus_1_power(x: Array | int, /) -> Array | int:
    return 1 - 2 * (x % 2)


def RS_all(
    kr: Array, theta: Array, phi: Array, *, n_end: int, type: Literal["regular", "singular"]
) -> Array:
    r"""Regular / Singular elementary solution of 3D Helmholtz equation.

    $$
    Y_n^m (\theta, \phi) :=
    (-1)^m \sqrt{\frac{(2n+1)(n-\abs{m})!}{4 \pi (n+\abs{m})!}}
    P_n^{\abs{m}} (\cos \theta) e^{i m \phi}
    $$

    $$
    R_n^m (kr, \theta, \phi) := j_n(kr) Y_n^m (\theta, \phi)
    $$

    $$
    S_n^m (kr, \theta, \phi) := h_n^{(1)}(kr) Y_n^m (\theta, \phi)
    $$

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
        Regular / Singular elementary solution of
        3D Helmholtz equation of shape (..., n_end**2)
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
    tmp = spherical_jn(n, kr[..., None])
    if type == "singular":
        tmp = tmp + 1j * spherical_yn(n, kr[..., None])
    return xp.asarray(
        minus_1_power((xp.abs(m) - m) // 2)
        * tmp
        * np.moveaxis(sph_harm_y_all(n_end - 1, n_end - 1, theta, phi)[n, m, ...], 0, -1),
        dtype=dtype,
        device=device,
    )
