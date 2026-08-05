from numba import njit
from numpy import abs as _abs
from numpy import sqrt
from .algebra import dot_scalar

# Projection statistics

_JIT = dict(nogil=True, fastmath=True, cache=True, inline="always")


@njit(**_JIT)
def mean_along_axis(
    x, y, z,
    ax, ay, az,
):
    """
    Mean projection along a (normalized) axis.
    """

    n = x.size

    s = 0.0

    for i in range(n):
        s += dot_scalar(
            x[i], y[i], z[i],
            ax, ay, az,
        )

    return s / n


@njit(**_JIT)
def variance_along_axis(
    x, y, z,
    ax, ay, az,
):
    """
    Population variance of projections along a (normalized) axis.
    """

    n = x.size

    mean = mean_along_axis(
        x, y, z,
        ax, ay, az,
    )

    var = 0.0

    for i in range(n):

        d = (
            dot_scalar(
                x[i], y[i], z[i],
                ax, ay, az,
            )
            - mean
        )

        var += d * d

    return var / n


@njit(**_JIT)
def std_along_axis(
    x, y, z,
    ax, ay, az,
):
    """
    Standard deviation along a (normalized) axis.
    """

    return sqrt(
        variance_along_axis(
            x, y, z,
            ax, ay, az,
        )
    )


@njit(**_JIT)
def minmax_along_axis(
    x, y, z,
    ax, ay, az,
):
    """
    Minimum and maximum projections.
    """

    n = x.size

    p = dot_scalar(
        x[0], y[0], z[0],
        ax, ay, az,
    )

    pmin = p
    pmax = p

    for i in range(1, n):

        p = dot_scalar(
            x[i], y[i], z[i],
            ax, ay, az,
        )

        if p < pmin:
            pmin = p
        elif p > pmax:
            pmax = p

    return pmin, pmax


@njit(**_JIT)
def extent_along_axis(
    x, y, z,
    ax, ay, az,
):
    """
    Extent along a (normalized) axis.
    """

    pmin, pmax = minmax_along_axis(
        x, y, z,
        ax, ay, az,
    )

    return pmax - pmin


@njit(**_JIT)
def rms_along_axis(
    x, y, z,
    ax, ay, az,
):
    """
    Root-mean-square projection along an axis.
    """

    n = x.size

    s = 0.0

    for i in range(n):

        p = dot_scalar(
            x[i], y[i], z[i],
            ax, ay, az,
        )

        s += p * p

    return sqrt(s / n)


@njit(**_JIT)
def mean_absolute_along_axis(
    x, y, z,
    ax, ay, az,
):
    """
    Mean absolute projection along an axis.
    """

    n = x.size

    s = 0.0

    for i in range(n):

        p = dot_scalar(
            x[i], y[i], z[i],
            ax, ay, az,
        )

        s += _abs(p)

    return s / n


@njit(**_JIT)
def projection_moments(
    x, y, z,
    ax, ay, az,
):
    """
    Return mean, variance, standard deviation,
    minimum and maximum projection.
    """

    pmin, pmax = minmax_along_axis(
        x, y, z,
        ax, ay, az,
    )

    mean = mean_along_axis(
        x, y, z,
        ax, ay, az,
    )

    n = x.size

    var = 0.0

    for i in range(n):

        d = (
            dot_scalar(
                x[i], y[i], z[i],
                ax, ay, az,
            )
            - mean
        )

        var += d * d

    var /= n

    return (
        mean,
        var,
        sqrt(var),
        pmin,
        pmax,
    )
