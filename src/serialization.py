"""JSON helpers for NumPy values."""

import numpy as np


def json_default(value: object) -> object:
    """Convert NumPy values for durable JSON output.

    Parameters
    ----------
    value : object
        NumPy array or scalar encountered by the JSON encoder.

    Returns
    -------
    object
        Native list or scalar.

    Raises
    ------
    TypeError
        If the value has an unsupported type.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(str(type(value)))
