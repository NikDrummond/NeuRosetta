# pca.py

import numpy as np


def principal_components(coords):
    mean = np.mean(coords, axis=0)
    _, _, result = np.linalg.svd(coords - mean)
    return result


def major_axis(coords):
    return principal_components(coords)[0]
