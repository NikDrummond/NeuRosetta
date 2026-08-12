from numpy import mean, empty, float64, eye, sqrt, array, ndarray
from numpy.linalg import inv, eigh
from numba import njit

from ._validation import check

_JIT = dict(nogil=True, fastmath=True, cache=True, inline="always")


@njit(**_JIT)
def covariance_xyz(x, y, z):
    """
    Compute the unbiased 3x3 covariance matrix from x, y, z coordinates.

    Parameters
    ----------
    x, y, z : 1D float arrays of equal length

    Returns
    -------
    cov : (3,3) ndarray
    """
    n = x.size

    mx = mean(x)
    my = mean(y)
    mz = mean(z)

    cxx = 0.0
    cxy = 0.0
    cxz = 0.0
    cyy = 0.0
    cyz = 0.0
    czz = 0.0

    for i in range(n):
        dx = x[i] - mx
        dy = y[i] - my
        dz = z[i] - mz

        cxx += dx * dx
        cxy += dx * dy
        cxz += dx * dz
        cyy += dy * dy
        cyz += dy * dz
        czz += dz * dz

    s = 1.0 / (n - 1)

    cov = empty((3, 3), dtype=float64)

    cov[0, 0] = cxx * s
    cov[0, 1] = cxy * s
    cov[0, 2] = cxz * s

    cov[1, 0] = cov[0, 1]
    cov[1, 1] = cyy * s
    cov[1, 2] = cyz * s

    cov[2, 0] = cov[0, 2]
    cov[2, 1] = cov[1, 2]
    cov[2, 2] = czz * s

    return cov


@njit(**_JIT)
def robust_covariance_xyz(
    x,
    y,
    z,
    c=1.5,
    tol=1e-6,
    max_iter=100,
):
    """
    Robust covariance using a Huber M-estimator.

    Parameters
    ----------
    x, y, z : 1D arrays
        Input coordinates.
    c : float
        Huber tuning constant.
    tol : float
        Convergence tolerance on centroid displacement.
    max_iter : int
        Maximum IRLS iterations.

    Returns
    -------
    cov : (3,3) ndarray
        Robust covariance matrix.
    """

    n = x.size

    # Initial centroid
    mx = mean(x)
    my = mean(y)
    mz = mean(z)

    # Initial covariance
    cov = covariance_xyz(x, y, z)

    # Reused every iteration
    weights = empty(n, dtype=float64)

    c2 = c * c

    for _ in range(max_iter):

        # Invert covariance matrix
        inv_cov = inv(cov + eye(3) * 1e-6)

        #
        # Step 1: compute Huber weights
        #
        wsum = 0.0

        for i in range(n):

            dx = x[i] - mx
            dy = y[i] - my
            dz = z[i] - mz

            md2 = (
                dx * (inv_cov[0, 0] * dx + inv_cov[0, 1] * dy + inv_cov[0, 2] * dz)
                + dy * (inv_cov[1, 0] * dx + inv_cov[1, 1] * dy + inv_cov[1, 2] * dz)
                + dz * (inv_cov[2, 0] * dx + inv_cov[2, 1] * dy + inv_cov[2, 2] * dz)
            )

            if md2 < c2:
                w = 1.0
            else:
                w = c2 / md2

            weights[i] = w
            wsum += w

        #
        # Step 2: weighted centroid
        #
        mx_new = 0.0
        my_new = 0.0
        mz_new = 0.0

        for i in range(n):
            w = weights[i]

            mx_new += w * x[i]
            my_new += w * y[i]
            mz_new += w * z[i]

        mx_new /= wsum
        my_new /= wsum
        mz_new /= wsum

        #
        # Check convergence
        #
        diff = sqrt((mx_new - mx) ** 2 + (my_new - my) ** 2 + (mz_new - mz) ** 2)

        mx = mx_new
        my = my_new
        mz = mz_new

        #
        # Step 3: weighted covariance using stored weights
        #
        cxx = 0.0
        cxy = 0.0
        cxz = 0.0
        cyy = 0.0
        cyz = 0.0
        czz = 0.0

        for i in range(n):

            w = weights[i]

            dx = x[i] - mx
            dy = y[i] - my
            dz = z[i] - mz

            cxx += w * dx * dx
            cxy += w * dx * dy
            cxz += w * dx * dz
            cyy += w * dy * dy
            cyz += w * dy * dz
            czz += w * dz * dz

        inv_wsum = 1.0 / wsum

        cov[0, 0] = cxx * inv_wsum
        cov[0, 1] = cxy * inv_wsum
        cov[0, 2] = cxz * inv_wsum

        cov[1, 0] = cov[0, 1]
        cov[1, 1] = cyy * inv_wsum
        cov[1, 2] = cyz * inv_wsum

        cov[2, 0] = cov[0, 2]
        cov[2, 1] = cov[1, 2]
        cov[2, 2] = czz * inv_wsum

        if diff < tol:
            break

    return cov


@njit(**_JIT)
def covariance_eigh(cov):
    """
    Eigenvalues and eigenvectors of a symmetric covariance matrix.

    Returns
    -------
    eigenvalues : (3,)
        Descending order.
    eigenvectors : (3,3)
        Columns correspond to eigenvalues.
    """

    vals, vecs = eigh(cov)

    vals = vals[::-1]
    vecs = vecs[:, ::-1]

    # Optional: enforce right-handed basis
    v0 = vecs[:, 0]
    v1 = vecs[:, 1]
    v2 = vecs[:, 2]

    cross = array(
        (
            v0[1] * v1[2] - v0[2] * v1[1],
            v0[2] * v1[0] - v0[0] * v1[2],
            v0[0] * v1[1] - v0[1] * v1[0],
        )
    )

    if cross[0] * v2[0] + cross[1] * v2[1] + cross[2] * v2[2] < 0.0:
        vecs[:, 2] *= -1.0

    return vals, vecs


def eig_decomp(x: ndarray, y: ndarray, z: ndarray, robust: bool = False, norm=False):
    """
    Eigen-decomposition of the covariance matrix of 3D coordinates.

    Computes principal axes and explained variance from ``x``, ``y``, and ``z``
    coordinate arrays via covariance estimation and symmetric eigendecomposition.

    Parameters
    ----------
    x, y, z : ndarray
        1D arrays of equal length containing point coordinates.
    robust : bool, default False
        If True, estimate covariance with a Huber M-estimator instead of the
        sample covariance.
    norm : bool, default False
        If True, normalize eigenvalues to sum to 1 (fraction of total variance).

    Returns
    -------
    evals : (3,) ndarray
        Eigenvalues in descending order.
    evecs : (3, 3) ndarray
        Eigenvectors as columns, aligned with ``evals``. The basis is
        right-handed.
    """
    check(locals(), "x", (-1,))
    check(locals(), "y", (-1,))
    check(locals(), "z", (-1,))

    # egt cov mat
    if robust:
        cov = robust_covariance_xyz(x, y, z)
    else:
        cov = covariance_xyz(x, y, z)

    # eig decomp
    evals, evecs = covariance_eigh(cov)

    if norm:
        evals = evals / evals.sum()

    return evals, evecs
