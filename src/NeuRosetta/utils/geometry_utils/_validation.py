## _validation.py

import numpy as np


def pluralize(noun, count):
    """
    Return a noun in singular or plural form.

    Parameters
    ----------
    noun : str
        Singular noun.
    count : int
        Count determining plurality.

    Returns
    -------
    str
        ``noun`` if ``count == 1``, otherwise ``f"{noun}s"``.
    """
    return noun if count == 1 else f"{noun}s"


def raise_dimension_error(*input_values):
    """
    Raise a ``ValueError`` for unsupported array dimensionalities.

    Parameters
    ----------
    *input_values : array-like
        Inputs whose ``ndim`` values are reported in the error message.

    Raises
    ------
    ValueError
        Always raised with a message describing the input dimensionalities.
    """
    messages = [
        f"{input_value.ndim} {pluralize('dimension', input_value.ndim)}"
        for input_value in input_values
    ]
    if len(messages) == 1:
        message = messages[0]
    elif len(messages) == 2:
        message = f"{messages[0]} and {messages[1]}"
    else:
        message = "those inputs"
    raise ValueError(f"Not sure what to do with {message}")


def check_value(arr, shape, name=None):
    """
    Validate that an array has an expected shape.

    Wildcard dimensions are denoted by ``-1``.

    Parameters
    ----------
    arr : array-like
        Array to validate.
    shape : tuple of int
        Expected shape. Use ``-1`` for dimensions that may vary.
    name : str, optional
        Parameter name used in error messages.

    Returns
    -------
    int or tuple of int or None
        Size(s) of wildcard dimension(s), or ``None`` if there are no
        wildcards.

    Raises
    ------
    ValueError
        If ``arr`` is ``None``, not array-like, or has the wrong shape.
    """

    def is_wildcard(dim):
        return dim == -1

    if any(not isinstance(dim, int) and not is_wildcard(dim) for dim in shape):
        raise ValueError("Expected shape dimensions to be int")

    if name is None:
        preamble = "Expected an array"
    else:
        preamble = f"{name} must be an array"

    if arr is None:
        raise ValueError(f"{preamble} with shape {shape}; got None")
    try:
        len(arr.shape)
    except (AttributeError, TypeError):
        raise ValueError(f"{preamble} with shape {shape}; got {arr.__class__.__name__}")

    if len(arr.shape) != len(shape) or any(
        actual != expected
        for actual, expected in zip(arr.shape, shape)
        if not is_wildcard(expected)
    ):
        raise ValueError(f"{preamble} with shape {shape}; got {arr.shape}")

    wildcard_dims = [
        actual for actual, expected in zip(arr.shape, shape) if is_wildcard(expected)
    ]
    if len(wildcard_dims) == 0:
        return None
    elif len(wildcard_dims) == 1:
        return wildcard_dims[0]
    else:
        return tuple(wildcard_dims)


def check_value_any(arr, *shapes, name=None):
    """
    Validate that an array matches one of several expected shapes.

    Parameters
    ----------
    arr : array-like
        Array to validate.
    *shapes : tuple of int
        Accepted shapes. Each shape may contain ``-1`` wildcards.
    name : str, optional
        Parameter name used in error messages.

    Returns
    -------
    int or tuple of int or None
        Wildcard size(s) from the first matching shape, or ``None`` if there
        are no wildcards.

    Raises
    ------
    ValueError
        If no ``shapes`` are provided, or if ``arr`` matches none of them.
    """

    if len(shapes) == 0:
        raise ValueError("At least one shape is required")
    for shape in shapes:
        try:
            return check_value(arr, shape, name=name or "arr")
        except ValueError:
            pass

    if name is None:
        preamble = "Expected an array"
    else:
        preamble = f"Expected {name} to be an array"

    if len(shapes) == 1:
        (shape_choices,) = shapes
    else:
        shape_choices = ", ".join(
            [str(s) for s in shapes[:-2]]
            + [" or ".join([str(shapes[-2]), str(shapes[-1])])]
        )

    if arr is None:
        raise ValueError(f"{preamble} with shape {shape_choices}; got None")
    else:
        try:
            len(arr.shape)
        except (AttributeError, TypeError):
            raise ValueError(
                f"{preamble} with shape {shape_choices}; got {arr.__class__.__name__}"
            )
        raise ValueError(f"{preamble} with shape {shape_choices}; got {arr.shape}")


def check(locals_namespace, name, shape):
    """
    Validate a local variable against an expected shape.

    Convenience wrapper around :func:`check_value` for use inside functions
    that already hold arguments in a ``locals()`` namespace.

    Parameters
    ----------
    locals_namespace : dict
        Mapping returned by ``locals()``.
    name : str
        Name of the variable to validate.
    shape : tuple of int
        Expected shape passed to :func:`check_value`.

    Returns
    -------
    int or tuple of int or None
        Wildcard size(s) from :func:`check_value`.
    """
    return check_value(locals_namespace[name], shape, name=name)


def columnize(arr, shape=(-1, 3), name=None):
    """
    Ensure an array has a columnar shape, reshaping if needed.

    Parameters
    ----------
    arr : array-like
        Input array.
    shape : tuple of int, default (-1, 3)
        Target shape with at least two dimensions. The array must match this
        shape or ``shape[1:]``.
    name : str, optional
        Parameter name used in validation error messages.

    Returns
    -------
    arr : ndarray
        Input array, reshaped when necessary.
    was_already_columnar : bool
        ``True`` if ``arr`` already had the target dimensionality.
    restore : callable
        Function that undoes a reshape when ``was_already_columnar`` is
        False.
    """
    if not isinstance(shape, tuple):
        raise ValueError("shape should be a tuple")
    if len(shape) < 2:
        raise ValueError("shape should have at least two dimensions")

    check_value_any(arr, shape, shape[1:], name=name or "arr")

    if arr.ndim == len(shape):
        return arr, True, lambda x: x
    else:
        return arr.reshape(*shape), False, lambda x: x[0]


### Check
def _check_scalar(x, y, z):
    """
    Check whether three vector components represent a scalar 3D vector.

    Parameters
    ----------
    x, y, z : scalar or array-like
        Components of a 3D vector.

    Returns
    -------
    bool
        True if all components are scalar (shape ``()``).

    Raises
    ------
    ValueError
        If the components do not share the same shape.
    """
    v_shapes = [np.shape(v) for v in (x, y, z)]

    if not (v_shapes[0] == v_shapes[1] == v_shapes[2]):
        raise ValueError(f"vector components have inconsistent shapes: {v_shapes}")

    return v_shapes[0] == v_shapes[1] == v_shapes[2] == ()


def _check_vector_broadcast(
    x1,
    y1,
    z1,
    x2,
    y2,
    z2,
    x3=None,
    y3=None,
    z3=None,
):
    """
    Check whether two or three vectors require broadcasting.

    Returns
    -------
    bool
        True if at least one vector must be broadcast.

    Raises
    ------
    ValueError
        If components within a vector have inconsistent shapes.
        If array vectors have incompatible shapes.
    """

    vectors = [
        (x1, y1, z1),
        (x2, y2, z2),
    ]

    if x3 is not None:
        vectors.append((x3, y3, z3))

    shapes = []

    for i, (x, y, z) in enumerate(vectors, start=1):

        s = (np.shape(x), np.shape(y), np.shape(z))

        if not (s[0] == s[1] == s[2]):
            raise ValueError(f"Vector {i} components have inconsistent shapes: {s}")

        shapes.append(s[0])

    scalar = [s == () for s in shapes]

    array_shapes = [s for s in shapes if s != ()]

    if len(array_shapes) > 1:

        first = array_shapes[0]

        for s in array_shapes[1:]:

            if s != first:
                raise ValueError(f"Vector shapes do not match: {array_shapes}")

    return any(scalar) and not all(scalar)


def _broadcast_vectors(
    x1,
    y1,
    z1,
    x2,
    y2,
    z2,
    x3=None,
    y3=None,
    z3=None,
):
    """
    Broadcast two or three vectors.

    Returns
    -------
    Tuple of broadcast-compatible vectors.
    """

    vectors = [
        [x1, y1, z1],
        [x2, y2, z2],
    ]

    have_third = x3 is not None

    if have_third:
        vectors.append([x3, y3, z3])

    #
    # Find target shape
    #

    target_shape = None

    for x, y, z in vectors:

        if np.ndim(x) > 0:

            target_shape = np.shape(x)
            break

    #
    # Nothing to do
    #

    if target_shape is None:

        if have_third:
            return (
                x1,
                y1,
                z1,
                x2,
                y2,
                z2,
                x3,
                y3,
                z3,
            )

        return (
            x1,
            y1,
            z1,
            x2,
            y2,
            z2,
        )

    #
    # Broadcast scalars
    #

    for v in vectors:

        for i in range(3):

            if np.ndim(v[i]) == 0:
                v[i] = np.full(target_shape, v[i], dtype=np.float64)

    if have_third:

        return (
            vectors[0][0],
            vectors[0][1],
            vectors[0][2],
            vectors[1][0],
            vectors[1][1],
            vectors[1][2],
            vectors[2][0],
            vectors[2][1],
            vectors[2][2],
        )

    return (
        vectors[0][0],
        vectors[0][1],
        vectors[0][2],
        vectors[1][0],
        vectors[1][1],
        vectors[1][2],
    )
