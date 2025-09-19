from pathlib import Path
from typing import Literal

import array_api_extra as xpx
import pytest
from array_api._2024_12 import Array, ArrayNamespaceFull
from array_api_compat import array_namespace
from array_api_compat import numpy as np
from array_api_negative_index import arange_asymmetric

from gumerov_expansion_coefficients._elementary_solutions import RS_all, idx_all, minus_1_power
from gumerov_expansion_coefficients._main import (
    idx,
    idx_i,
    ndim_harm,
    translational_coefficients,
    translational_coefficients_sectorial_init,
)


def euclidean_to_spherical(x: Array, y: Array, z: Array) -> tuple[Array, Array, Array]:
    xp = array_namespace(x)
    r = (x**2 + y**2 + z**2) ** 0.5
    theta = xp.atan2(xp.sqrt(x**2 + y**2), z)
    phi = xp.atan2(y, x)
    return r, theta, phi


def test_idx(xp: ArrayNamespaceFull) -> None:
    n = xp.asarray([0, 1, 1, 1])
    m = xp.asarray([0, 0, 1, -1])
    assert xp.all(idx(n, m) == xp.asarray([0, 1, 2, 3]))


def test_idx_i() -> None:
    assert idx_i(0, 0) == 0
    assert idx_i(1, -1) == 3
    assert idx_i(1, 0) == 1
    assert idx_i(1, 1) == 2


def test_idx_all(xp: ArrayNamespaceFull) -> None:
    n, m = idx_all(3, xp=xp, dtype=xp.int32, device=None)
    assert xp.all(n == xp.asarray([0, 1, 1, 1, 2, 2, 2, 2, 2]))
    assert xp.all(m == xp.asarray([0, 0, 1, -1, 0, 1, 2, -2, -1]))


def test_ndim_harm() -> None:
    assert ndim_harm(2) == 4


def test_minus_1_power() -> None:
    assert minus_1_power(0) == 1
    assert minus_1_power(1) == -1
    assert minus_1_power(2) == 1
    assert minus_1_power(3) == -1
    assert minus_1_power(4) == 1


def test_init(xp: ArrayNamespaceFull) -> None:
    # Gumerov (2, -7, 1)
    k = xp.asarray(1.0)
    t = xp.asarray([2, -7, 1])
    r, theta, phi = euclidean_to_spherical(t[0], t[1], t[2])
    assert r == pytest.approx(7.3484693)
    assert theta == pytest.approx(1.43429)
    assert phi == pytest.approx(-1.2924967)

    n_end = 4
    init = translational_coefficients_sectorial_init(k * r, theta, phi, True, n_end)
    assert init[idx_i(2, 1)] == pytest.approx(0.01413437 + 0.04947031j)
    assert init[idx_i(3, 2)] == pytest.approx(-0.01853696 + 0.01153411j)


def test_sectorial_n_m(xp: ArrayNamespaceFull) -> None:
    # Gumerov (2, -7, 1)
    k = xp.asarray(1.0)
    t = xp.asarray([2, -7, 1])
    r, theta, phi = euclidean_to_spherical(t[0], t[1], t[2])

    n_end = 4
    m = arange_asymmetric(n_end, xp=xp)
    sectorial = translational_coefficients(k * r, theta, phi, n_end=n_end, same=True)[
        :, idx(xp.abs(m), m)
    ]
    # assert sectorial[idx_i(1, 1), 0] == pytest.approx(0.01656551+0.05797928j)
    assert sectorial[idx_i(0, 0), 1] == pytest.approx(0.01656551 - 0.05797928j)
    assert sectorial[idx_i(0, 0), 2] == pytest.approx(0.15901178 + 0.09894066j)
    assert sectorial[idx_i(0, 0), 3] == pytest.approx(-0.04809683 + 0.04355622j)
    assert sectorial[idx_i(0, 0), -2] == pytest.approx(0.15901178 - 0.09894066j)
    assert sectorial[idx_i(1, 0), 1] == pytest.approx(-0.01094844 + 0.03831954j)
    assert sectorial[idx_i(1, -1), 1] == pytest.approx(-0.17418868 - 0.10838406j)
    assert sectorial[idx_i(1, 1), 1] == pytest.approx(0.18486702 + 0.0j)

    # assert sectorial[idx_i(2, 1), 0] == pytest.approx(-0.01413437 - 0.04947031j)
    assert sectorial[idx_i(2, 1), 1] == pytest.approx(-0.00290188 + 0.0j, abs=1e-7)
    assert sectorial[idx_i(2, 1), -1] == pytest.approx((0.01716189 - 0.01067851j), abs=1e-7)


def test_sectorial_nd_md(xp: ArrayNamespaceFull) -> None:
    # Gumerov (2, -7, 1)
    k = xp.asarray(1.0)
    t = xp.asarray([2, -7, 1])
    r, theta, phi = euclidean_to_spherical(t[0], t[1], t[2])

    n_end = 2
    coef = translational_coefficients(k * r, theta, phi, n_end=n_end, same=True)
    assert coef[idx_i(1, 1), idx_i(1, 0)] == coef[idx_i(1, 0), idx_i(1, -1)]
    assert coef[idx_i(1, 1), idx_i(1, 0)] == pytest.approx(-0.01094844 - 0.03831954j)


def test_main(xp: ArrayNamespaceFull) -> None:
    # Gumerov (2, -7, 1)
    k = xp.asarray(1.0)
    t = xp.asarray([2, -7, 1])
    r, theta, phi = euclidean_to_spherical(t[0], t[1], t[2])

    n_end = 5
    coef = translational_coefficients(
        k * r,
        theta,
        phi,
        same=True,
        n_end=n_end,
    )
    assert coef[idx_i(1, 0), idx_i(1, 0)] == pytest.approx(-0.01254681 + 0.0j)
    assert coef[idx_i(2, 1), idx_i(4, 3)] == pytest.approx(0.10999471 + 0.06844115j)
    assert coef[idx_i(2, 1), idx_i(4, -3)] == pytest.approx(-0.10065599 + 0.20439409j)
    assert coef[idx_i(2, -1), idx_i(4, -3)] == pytest.approx(0.10999471 - 0.06844115j)


@pytest.mark.parametrize("type", ["regular", "singular"])
def test_rs_all(xp: ArrayNamespaceFull, type: Literal["regular", "singular"]) -> None:
    k = xp.asarray(1.0)
    t = xp.asarray([2, -7, 1])
    r, theta, phi = euclidean_to_spherical(t[0], t[1], t[2])

    actual = RS_all(k * r, theta, phi, n_end=4, type=type)
    Path("tests/.cache").mkdir(exist_ok=True)
    np.savetxt(f"tests/.cache/{type}.csv", np.asarray(actual, dtype=np.complex128), delimiter=",")
    expected = xp.asarray(
        np.loadtxt(
            f"tests/{type}_Phase.CONDON_SHORTLEY.csv",
            delimiter=",",
            dtype=np.complex128,
        ),
        dtype=actual.dtype,
    )
    assert xp.all(xpx.isclose(actual, expected, atol=1e-6, rtol=1e-6))


@pytest.mark.parametrize("same", [True, False])
def test_main_all(xp: ArrayNamespaceFull, same: bool) -> None:
    k = xp.asarray(1.0)
    t = xp.asarray([2, -7, 1])
    r, theta, phi = euclidean_to_spherical(t[0], t[1], t[2])

    n_end = 3
    actual = translational_coefficients(
        k * r,
        theta,
        phi,
        same=same,
        n_end=n_end,
    ).T
    Path("tests/.cache").mkdir(exist_ok=True)
    np.savetxt(
        f"tests/.cache/translation_coef_{same}.csv",
        np.asarray(actual, dtype=np.complex128),
        delimiter=",",
    )
    expected = xp.asarray(
        np.loadtxt(
            f"tests/translation_coef_{same}_Phase.CONDON_SHORTLEY.csv",
            delimiter=",",
            dtype=np.complex128,
        ),
        dtype=actual.dtype,
    )
    assert xp.all(xpx.isclose(actual, expected, atol=1e-6, rtol=1e-6))


def test_gumerov_table(xp: ArrayNamespaceFull) -> None:
    k = 1.0

    x = xp.asarray([-1.0, 1.0, 0.0])
    t = xp.asarray([2.0, -7.0, 1.0])
    y = x + t

    # to spherical coordinates
    x_sp = euclidean_to_spherical(x[0], x[1], x[2])
    t_sp = euclidean_to_spherical(t[0], t[1], t[2])
    y_sp = euclidean_to_spherical(y[0], y[1], y[2])

    expected = {
        "exact": 0.049626 - 0.019882j,
        0: 0.020196 + 0.013655j,
        2: 0.049166 - 0.014885j,
        4: 0.049805 - 0.019548j,
        6: 0.049643 - 0.019875j,
        8: 0.049627 - 0.019883j,
    }

    y_S = RS_all(k * y_sp[0], y_sp[1], y_sp[2], n_end=6, type="singular")
    assert y_S[idx_i(5, 2)] == pytest.approx(expected["exact"], abs=1e-6)

    t_coef = translational_coefficients(k * t_sp[0], t_sp[1], t_sp[2], same=False, n_end=9)
    x_R = RS_all(k * x_sp[0], x_sp[1], x_sp[2], n_end=9, type="regular")
    for n_end in [9, 7, 5, 3, 1]:
        t_coef = t_coef[: ndim_harm(n_end)]
        y_S_sum = xp.sum(t_coef * x_R[: ndim_harm(n_end), None], axis=0)
        assert y_S_sum[idx_i(5, 2)] == pytest.approx(expected[n_end - 1], abs=1e-6)
