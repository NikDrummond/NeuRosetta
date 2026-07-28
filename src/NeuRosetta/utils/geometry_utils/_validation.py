## _validation.py

import numpy as np


def pluralize(noun, count):
    return noun if count == 1 else f"{noun}s"


def raise_dimension_error(*input_values):
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
    return check_value(locals_namespace[name], shape, name=name)


def columnize(arr, shape=(-1, 3), name=None):
    if not isinstance(shape, tuple):
        raise ValueError("shape should be a tuple")
    if len(shape) < 2:
        raise ValueError("shape should have at least two dimensions")

    check_value_any(arr, shape, shape[1:], name=name or "arr")

    if arr.ndim == len(shape):
        return arr, True, lambda x: x
    else:
        return arr.reshape(*shape), False, lambda x: x[0]