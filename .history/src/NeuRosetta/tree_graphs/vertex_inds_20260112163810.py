# functions to get node indicies from graphs

from numpy import ndarray, where, unique, concatenate

from ..core import _Tree
from typing import List
from ..errors.errors import _check_internal_property