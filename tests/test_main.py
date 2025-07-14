import pytest
from array_api._2024_12 import ArrayNamespaceFull

from gumerov_expansion_coefficients._main import (
    idx,
    idx_all,
    idx_i,
    minus_1_power,
    ndim_harm,
    translational_coefficients,
    translational_coefficients_sectorial_init,
    translational_coefficients_sectorial_n_m,
    translational_coefficients_sectorial_nd_md,
)


def test_idx(xp: ArrayNamespaceFull) -> None:
    n = xp.asarray([0, 1, 1, 1])
    m = xp.asarray([0, -1, 0, 1])
    assert xp.all(idx(n, m) == xp.asarray([0, 1, 2, 3]))


def test_idx_i() -> None:
    assert idx_i(0, 0) == 0
    assert idx_i(1, -1) == 1
    assert idx_i(1, 0) == 2
    assert idx_i(1, 1) == 3


def test_idx_all(xp: ArrayNamespaceFull) -> None:
    n, m = idx_all(3, xp=xp)
    assert xp.all(n == xp.asarray([0, 1, 1, 1, 2, 2, 2, 2, 2]))
    assert xp.all(m == xp.asarray([0, -1, 0, 1, -2, -1, 0, 1, 2]))


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
    r = xp.asarray(7.3484693)
    theta = xp.asarray(1.43429)
    phi = xp.asarray(-1.2924967)
    n_end = 4
    init = translational_coefficients_sectorial_init(k * r, theta, phi, True, n_end)
    assert init[idx_i(2, 1)] == pytest.approx(-0.01413437 - 0.04947031j)
    assert init[idx_i(3, 2)] == pytest.approx(-0.01853696 + 0.01153411j)


def test_sectorial_n_m(xp: ArrayNamespaceFull) -> None:
    # Gumerov (2, -7, 1)
    k = xp.asarray(1.0)
    r = xp.asarray(7.3484693)
    theta = xp.asarray(1.43429)
    phi = xp.asarray(-1.2924967)
    n_end = 3
    init = translational_coefficients_sectorial_init(k * r, theta, phi, True, n_end)
    sectorial = translational_coefficients_sectorial_n_m(
        n_end=n_end,
        translational_coefficients_sectorial_init=init,
    )
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
    r = xp.asarray(7.3484693)
    theta = xp.asarray(1.43429)
    phi = xp.asarray(-1.2924967)
    n_end = 2
    init = translational_coefficients_sectorial_init(k * r, theta, phi, True, n_end)
    sectorial_n_m = translational_coefficients_sectorial_n_m(
        n_end=n_end,
        translational_coefficients_sectorial_init=init,
    )
    sectorial_nd_md = translational_coefficients_sectorial_nd_md(
        n_end=n_end,
        translational_coefficients_sectorial_n_m=sectorial_n_m,
    )
    assert sectorial_nd_md[1, idx_i(1, 0)] == sectorial_n_m[idx_i(1, 0), -1]
    assert sectorial_nd_md[1, idx_i(1, 0)] == pytest.approx(0.01094844 + 0.03831954j)


def test_main(xp: ArrayNamespaceFull) -> None:
    # Gumerov (2, -7, 1)
    k = xp.asarray(1.0)
    r = xp.asarray(7.3484693)
    theta = xp.asarray(1.43429)
    phi = xp.asarray(-1.2924967)
    n_end = 5
    coefs = translational_coefficients(
        k * r,
        theta,
        phi,
        same=True,
        n_end=n_end,
    )
    assert coefs[idx_i(1, 0), idx_i(1, 0)] == pytest.approx(-0.01254681 + 0.0j)
    assert coefs[idx_i(2, 1), idx_i(4, 3)] == pytest.approx(0.10999471 + 0.06844115j)
    assert coefs[idx_i(2, 1), idx_i(4, -3)] == pytest.approx(-0.10065599 + 0.20439409j)
    assert coefs[idx_i(2, -1), idx_i(4, -3)] == pytest.approx(0.10999471 - 0.06844115j)
