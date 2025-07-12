from array_api._2024_12 import ArrayNamespaceFull

from gumerov_expansion_coefficients.main import R, idx, idx_all


def test_idx() -> None:
    assert idx(0, 0) == 0
    assert idx(1, -1) == 1
    assert idx(1, 0) == 2
    assert idx(1, 1) == 3


def test_idx_all(xp: ArrayNamespaceFull) -> None:
    n, m = idx_all(3, xp=xp)
    assert xp.all(n == xp.asarray([0, 1, 1, 1, 2, 2, 2, 2, 2]))
    assert xp.all(m == xp.asarray([0, -1, 0, 1, -2, -1, 0, 1, 2]))


def test_R(xp: ArrayNamespaceFull) -> None:
    m, n = idx_all(3, xp=xp)
    theta = xp.linspace(0, xp.pi, 10)
    phi = xp.linspace(0, 2 * xp.pi, 10)
    k = xp.ones_like(theta)
    result = R(n, m, theta[:, None], phi[None, :], k)
    assert result.shape == (10, 10, 9)  # 10 theta
