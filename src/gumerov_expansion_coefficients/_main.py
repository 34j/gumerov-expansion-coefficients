# https://github.com/search?q=gumerov+translation+language%3APython&type=code&l=Python
from math import cos, sin, sqrt
from typing import Any

import numba
from array_api._2024_12 import Array
from array_api_compat import array_namespace
from numba import complex64, complex128, float32, float64, jit, prange
from numba.cuda.cudadrv.error import CudaSupportError

from gumerov_expansion_coefficients._elementary_solutions import RS_all, idx_all, minus_1_power

# Gumerov's notation
# E^m_n = sum_{m'n'} (E|F)^{m' m}_{n' n} F^{m'}_{n'}
# (E|F)^{m' m}_{n'} := (E|F)^{m' m}_{n' |m|}
# (E|F)^{m' m}_{,n} := (E|F)^{m' m}_{|m'| n}


@jit()
def idx_i(n: int, m: int, /) -> int:
    """Index for the coefficients."""
    # (0, 0) -> 0
    # (1, -1) -> 1
    # (1, 0) -> 2
    return n**2 + (m % (2 * n + 1))


def idx(n: Array, m: Array, /) -> Array:
    """Index for the coefficients."""
    # (0, 0) -> 0
    # (1, -1) -> 1
    # (1, 0) -> 2
    xp = array_namespace(n, m)
    m_abs = xp.abs(m)
    return xp.where(m_abs > n, -1, n**2 + (m % (2 * n + 1)))


@jit(inline="always")
def ndim_harm(n_end: int, /) -> int:
    """Number of spherical harmonics which degree is less than n_end."""
    return n_end**2


minus_1_power_jit = jit(inline="always")(minus_1_power)


def translational_coefficients_sectorial_init(
    kr: Array, theta: Array, phi: Array, same: bool, n_end: int, /
) -> Array:
    """Initial values of sectorial translational coefficients $ (E|F)^{m',0}_{n', 0} $

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
    # 4.43 / 4.58
    # (E|F)^{m' 0}_{n' 0} = (E|F)^{m' 0}_{n'}
    return (
        minus_1_power(n)
        * xp.sqrt(xp.asarray(4.0, dtype=kr.dtype, device=kr.device) * xp.pi)
        * RS_all(kr, theta, phi, n_end=2 * n_end - 1, type="regular" if same else "singular")[
            ..., idx(n, -m)
        ]
    )


def rotational_coefficients_init(theta: Array, phi: Array, n_end: int, /) -> Array:
    """Initial values of rotational coefficients $ T^{m',0}_{n} $.

    Parameters
    ----------
    theta : Array
        polar angle of shape (...,)
    phi : Array
        azimuthal angle of shape (...,)
    n_end : int
        Maximum degree of spherical harmonics.

    Returns
    -------
    Array
        Initial rotational coefficients of shape (..., ndim_harm(n_end),)
    """
    xp = array_namespace(theta, phi)
    n, m = idx_all(2 * n_end - 1, xp=xp, dtype=xp.int32, device=theta.device)
    # 4.66
    return (
        xp.sqrt(xp.asarray(4.0, dtype=theta.dtype, device=theta.device) * xp.pi / (2 * n + 1))
        * RS_all(None, theta, phi, n_end=2 * n_end - 1, type="harmonics")[..., idx(n, -m)]
    )


@jit(inline="always")
def _set_coef(a: Array, nd: int, md: int, n: int, m: int, value: float, swap: bool = False) -> None:
    if swap:
        a[idx_i(n, m), idx_i(nd, md)] = value
    else:
        a[idx_i(nd, md), idx_i(n, m)] = value


@jit(inline="always")
def _get_coef(a: Array, nd: int, md: int, n: int, m: int, swap: bool = False) -> float:
    if swap:
        return a[idx_i(n, m), idx_i(nd, md)]
    else:
        return a[idx_i(nd, md), idx_i(n, m)]


def _translational_coefficients_all(
    translational_coefficients_sectorial_init: Array, ret: Array, _: Array, /
) -> None:
    """Translational coefficients $ (E|F)^{m',m}_{n',n} $

    Parameters
    ----------
    translational_coefficients_sectorial_init : Array
        Initial sectorial translational coefficients $ (E|F)^{m',0}_{n', 0} $
        of shape (..., ndim_harm(2 * n_end - 1),)
    ret : Array
        Empty array to store the result of shape (..., ndim_harm(n_end), ndim_harm(n_end))
    _ : Array
        Dummy return array for numba guvectorize

    """
    n_end = (int(sqrt(ret.shape[-1])) + 1) // 2
    for nd in prange(2 * n_end - 1):
        for md in prange(-nd, nd + 1):
            _set_coef(ret, nd, md, 0, 0, translational_coefficients_sectorial_init[idx_i(nd, md)])

    for m in range(2 * n_end - 2):
        n = abs(m)
        for nd in prange(2 * n_end - n - 2):
            for md in prange(-nd, nd + 1):
                tmp = -b(nd + 1, md - 1) * _get_coef(ret, nd + 1, md - 1, n, m)  # 3rd term
                if abs(md - 1) <= nd - 1:
                    tmp += b(nd, -md) * _get_coef(ret, nd - 1, md - 1, n, m)  # 4th term
                tmp /= b(n + 1, -m - 1)
                _set_coef(ret, nd, md, n + 1, m + 1, tmp)  # 2nd term

    for m in range(2 * n_end - 2):
        m = -m
        n = abs(m)
        for nd in prange(2 * n_end - n - 2):
            for md in prange(-nd, nd + 1):
                tmp = b(nd, md) * _get_coef(ret, nd - 1, md + 1, n, m)  # 1st term
                if abs(md + 1) <= nd + 1:
                    tmp -= b(nd + 1, -md - 1) * _get_coef(ret, nd + 1, md + 1, n, m)  # 2nd term
                tmp /= b(n + 1, m - 1)
                _set_coef(ret, nd, md, n + 1, m - 1, tmp)  # 3rd term

    for m in prange(-2 * n_end + 2, 2 * n_end - 1):
        n = abs(m)
        for nd in prange(2 * n_end - 1):
            for md in prange(-nd, nd + 1):
                _set_coef(
                    ret,
                    nd,
                    md,
                    n,
                    m,
                    float32(minus_1_power_jit(n + nd)) * _get_coef(ret, nd, -md, n, -m),
                    swap=True,
                )

    for m in prange(-n_end + 1, n_end):
        for md in prange(-n_end + 1, n_end):
            mabs, mdabs = abs(m), abs(md)
            mlarger = max(mabs, mdabs)
            n_iter = n_end - mlarger - 1

            for m1_is_md, m1, m2 in ((True, md, m), (False, m, md)):
                # del m, md
                m1abs = abs(m1)
                m2abs = abs(m2)
                for i in range(n_iter):
                    n2 = i + m2abs  # n
                    for n1 in prange(m1abs + i + 1, 2 * n_end - mlarger - i - 2):  # nd
                        tmp = (
                            -a(n1, m1) * _get_coef(ret, n1 + 1, m1, n2, m2, swap=m1_is_md)  # 3rd
                            + a(n1 - 1, m1)
                            * _get_coef(ret, n1 - 1, m1, n2, m2, swap=m1_is_md)  # 4th
                        )
                        if i > 0:
                            tmp += (
                                a(n2 - 1, m2)
                                * _get_coef(ret, n1, m1, n2 - 1, m2, swap=m1_is_md)  # 1st
                            )  # 1st
                        _set_coef(ret, n1, m1, n2 + 1, m2, tmp / a(n2, m2), swap=m1_is_md)  # 2nd


def n_rotational_coefficients(n_end: int) -> int:
    r"""Number of rotational coefficients with degree less than n_end.

    .. math::
        \sum_{n=0}^{N-1} (2n+1)^2 = \frac{N(2N-1)(2N+1)}{3}
    """
    return n_end * (2 * n_end - 1) * (2 * n_end + 1) // 3


def idx_rot(n: int, md: int, m: int, /) -> int:
    n_times_2_plus_1 = 2 * n + 1
    return (
        n_rotational_coefficients(n)
        + n_times_2_plus_1 * (md % n_times_2_plus_1)
        + (m % n_times_2_plus_1)
    )


_impl = {}

for dtype_f, dtype_c in ((float32, complex64), (float64, complex128)):

    @jit()
    def a(n: int, m: int, /) -> float:
        m_abs = abs(m)
        if m_abs > n:
            return dtype_f(0)  # noqa: B023
        return sqrt(dtype_f((n + m_abs + 1) * (n - m_abs + 1)) / dtype_f((2 * n + 1) * (2 * n + 3)))  # noqa: B023

    @jit()
    def b(n: int, m: int, /) -> float:
        m_abs = abs(m)
        if m_abs > n:
            return dtype_f(0)  # noqa: B023
        tmp = sqrt(dtype_f((n - m - 1) * (n - m)) / dtype_f((2 * n - 1) * (2 * n + 1)))  # noqa: B023
        if m >= 0:
            return tmp
        else:
            return -tmp

    def _rotational_coefficients_all(
        rotational_coefficients_init: Array,
        theta: Array,
        phi: Array,
        xi: Array,
        ret: Array,
        _: Array,
        /,
    ) -> None:
        """Rotational coefficients $ T^{m',m}_{n'=n} $

        Parameters
        ----------
        rotational_coefficients_init : Array
            Initial rotational coefficients $ T^{m',0}_{n} $
            of shape (..., ndim_harm(2 * n_end - 1),)
        theta : Array
            polar angle of shape (...,)
        phi : Array
            azimuthal angle of shape (...,)
        xi : Array
            azimuthal angle of shape (...,)
        ret : Array
            Empty array to store the result of shape (..., (N_end-1)(N_end)(2*N_end-1) / 6)
            where N_end = 2 * n_end - 1
        _ : Array
            Dummy return array for numba guvectorize
        """
        n_end = (ret.shape[-3] + 1) // 2
        for n in prange(2 * n_end - 1):
            for md in prange(-n, n + 1):
                ret[idx_rot(n, md, 0)] = rotational_coefficients_init[idx_i(n, md)]

        one_f = dtype_f(1)  # noqa: B023
        one_j = dtype_c(1j)  # noqa: B023
        if phi == 0:
            exp_phi = one_f
            exp_phi_conj = one_f
        else:
            exp_phi = cos(phi) + one_j * sin(phi)
            exp_phi_conj = cos(phi) - one_j * sin(phi)
        if xi == 0:
            exp_xi = one_f
        else:
            exp_xi = cos(xi) + one_j * sin(xi)
        one_minus_cos_theta = exp_phi * (one_f - cos(theta))
        one_plus_cos_theta = exp_phi_conj * (one_f + cos(theta))
        sin_theta = sin(theta)
        for m in range(n_end - 1):
            for n in prange(2 + m, 2 * n_end - 1 - m):
                for md in prange(-n + 1, n):
                    ret[idx_rot(n - 1, md, m + 1)] = (
                        exp_xi
                        / (b(n, m))
                        * (
                            1
                            / 2
                            * (
                                b(n, -md - 1) * one_minus_cos_theta * ret[idx_rot(n, md + 1, m)]
                                - b(n, md - 1) * one_plus_cos_theta * ret[idx_rot(n, md - 1, m)]
                            )
                            - a(n - 1, md) * sin_theta * ret[idx_rot(n, md, m)]
                        )
                    )
        for n in prange(2, n_end):
            for md in prange(-n, n + 1):
                for m in prange(n + 1):
                    ret[idx_rot(n, -md, -m)] = ret[idx_rot(n, md, m)]

    _args: dict[str, tuple[list[Any], str]] = {
        "translational": (
            [(dtype_c[:], dtype_c[:, :], dtype_c)],
            "(n),(n,n)->()",
        ),
        "rotational": (
            [
                (dtype_f[:], dtype_f, dtype_f, dtype_f, dtype_f, dtype_f),
                (dtype_c[:], dtype_f, dtype_f, dtype_f, dtype_c, dtype_f),
            ],
            "(n),( ),( ),( ),(m)->()",
        ),
    }
    for k in ["translational", "rotational"]:
        _impl[(k, "parallel", dtype_c)] = numba.guvectorize(*_args[k], target="parallel")(
            _translational_coefficients_all
        )

        try:
            _impl[(k, "cuda", dtype_c)] = numba.guvectorize(
                *_args[k],
                target="cuda",
            )(_translational_coefficients_all)
        except CudaSupportError:
            _impl[(k, "cuda", dtype_c)] = None


def translational_coefficients_all(
    *,
    translational_coefficients_sectorial_init: Array,
) -> Array:
    """Translational coefficients (E|F)^{m',m}_{n',n}

    Parameters
    ----------
    translational_coefficients_sectorial_init : Array
        Initial sectorial translational coefficients (E|F)^{m',0}_{n', 0}
        of shape (..., ndim_harm(n_end),)

    Returns
    -------
    Array
        Translational coefficients [(m',n'),(m,n)] of shape (ndim_harm(n_end), ndim_harm(n_end))
    """
    xp = array_namespace(translational_coefficients_sectorial_init)
    dtype = translational_coefficients_sectorial_init.dtype
    device = translational_coefficients_sectorial_init.device
    shape = translational_coefficients_sectorial_init.shape[:-1]
    n_end = (int(sqrt(translational_coefficients_sectorial_init.shape[-1])) + 1) // 2
    ret = xp.zeros(
        (*shape, ndim_harm(2 * n_end - 1), ndim_harm(2 * n_end - 1)), dtype=dtype, device=device
    )
    _impl[
        (
            "translational",
            "cuda" if "cuda" in str(device) else "parallel",
            dtype,
        )
    ](
        translational_coefficients_sectorial_init,
        ret,
    )
    return xp.asarray(
        ret,
        dtype=dtype,
        device=device,
    )[: ndim_harm(n_end), : ndim_harm(n_end)]


def rotational_coefficients_all(
    *,
    rotational_coefficients_init: Array,
    theta: Array,
    phi: Array,
    xi: Array,
) -> tuple[Array, ...]:
    """Rotational coefficients $ T^{m',m}_{n'=n} $

    Parameters
    ----------
    rotational_coefficients_init : Array
        Initial rotational coefficients $ T^{m',0}_{n} $
        of shape (..., ndim_harm(2 * n_end - 1),)
    theta : Array
        polar angle of shape (...,)
    phi : Array
        azimuthal angle of shape (...,)
    xi : Array
        azimuthal angle of shape (...,)

    Returns
    -------
    tuple[Array, ...]
        Rotational coefficients $ T^{m',m}_{n'=n} $
        of shape (..., 2n+1, 2n+1) for n in [0, n_end-1]
    """
    xp = array_namespace(rotational_coefficients_init, theta, phi, xi)
    dtype = rotational_coefficients_init.dtype
    device = rotational_coefficients_init.device
    shape = rotational_coefficients_init.shape[:-1]
    n_end = (int(sqrt(rotational_coefficients_init.shape[-1])) + 1) // 2
    ret = xp.zeros(
        (*shape, n_rotational_coefficients(2 * n_end - 1)),
        dtype=dtype,
        device=device,
    )
    _impl[("rotational", "cuda" if "cuda" in str(device) else "parallel", dtype)](
        rotational_coefficients_init,
        theta,
        phi,
        xi,
        ret,
    )
    ret = xp.asarray(
        ret,
        dtype=dtype,
        device=device,
    )
    result = []
    for n in range(n_end):
        result.append(
            ret[..., n_rotational_coefficients(n) : n_rotational_coefficients(n + 1)].reshape(
                *shape, 2 * n + 1, 2 * n + 1
            )
        )
    return tuple(result)


def translational_coefficients(
    kr: Array, theta: Array, phi: Array, *, same: bool, n_end: int
) -> Array:
    r"""Translational coefficients (E|F)^{m',m}_{n',n}.

    .. math::
        Y_n^m (\theta, \phi) &:=
        (-1)^m \sqrt{\frac{(2n+1)(n-\left|m\right|)!}{4 \pi (n+\left|m\right|)!}}
        P_n^{\left|m\right|} (\cos \theta) e^{i m \phi}

        R_n^m (kr, \theta, \phi) &:= j_n(kr) Y_n^m (\theta, \phi)

        S_n^m (kr, \theta, \phi) &:= h_n^{(1)}(kr) Y_n^m (\theta, \phi)

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
        Initial sectorial translational coefficients of shape (..., n_end**2)
    """
    translational_coefficients_sectorial_init_ = translational_coefficients_sectorial_init(
        kr, theta, phi, same, n_end
    )
    return translational_coefficients_all(
        translational_coefficients_sectorial_init=translational_coefficients_sectorial_init_,
    )


def rotational_coefficients(
    theta: Array, phi: Array | None, xi: Array | None, *, n_end: int
) -> tuple[Array, ...]:
    r"""Rotational coefficients $ T^{m',m}_{n'=n} $.

    .. math::
        Y_n^m (\theta, \phi) &:=
        (-1)^m \sqrt{\frac{(2n+1)(n-\left|m\right|)!}{4 \pi (n+\left|m\right|)!}}
        P_n^{\left|m\right|} (\cos \theta) e^{i m \phi}

    Parameters
    ----------
    theta : Array
        polar angle of shape (...,)
    phi : Array | None
        azimuthal angle of shape (...,). If None, treated as 0.
    xi : Array | None
        azimuthal angle of shape (...,). If None, treated as 0.
    n_end : int
        Maximum degree of spherical harmonics.

    Returns
    -------
    Array
        Rotational coefficients $ T^{m',m}_{n'=n} $
        of shape (..., 2n+1, 2n+1) for n in [0, n_end-1]
    """
    xp = array_namespace(theta, phi, xi)
    if phi is None and xi is None:
        is_real = True
        phi = xp.zeros_like(theta)
        xi = xp.zeros_like(theta)
    elif phi is not None and xi is not None:
        is_real = False
    else:
        raise ValueError("phi and xi must be both None or both not None.")
    rotational_coefficients_init_ = rotational_coefficients_init(theta, phi, n_end)
    if is_real:
        rotational_coefficients_init_ = xp.real(rotational_coefficients_init_)
    return rotational_coefficients_all(
        rotational_coefficients_init=rotational_coefficients_init_,
        theta=theta,
        phi=phi,
        xi=xi,
    )
