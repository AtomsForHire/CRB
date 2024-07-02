import datetime
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
from mwa_qa import read_metafits

from misc import print_with_time


def partial(u, v, source_list, gain):
    """Function for evaluating the partial derivative expression

    Parameters
    ----------
    u: float
        baseline coord
    v: float
        baseline coord
    source_list: array
        source coordinates and intensity (l, m, B)
    gain: float
        the gain of the antenna
    """

    result = 0
    for i in range(0, len(source_list)):
        result += source_list[i, 2] * np.exp(
            -2.0 * np.pi * 1j * (u * source_list[i, 0] + v * source_list[i, 1])
        )

    return result * gain


def partial_star(u, v, source_list, gain):
    """Function for evaluating the complex conjugate partial derivative expression

    Parameters
    ----------
    u: float
        baseline coord
    v: float
        baseline coord
    source_list: array
        source coordinates and intensity (l, m, B)
    gain: float
        the gain of the antenna
    """
    result = 0
    for i in range(0, len(source_list)):
        result += source_list[i, 2] * np.exp(
            2.0 * np.pi * 1j * (u * source_list[i, 0] + v * source_list[i, 1])
        )

    return result * gain


def propagate(baselines, source_list, uncertainties):
    """Function for propagating gain errors into visibilities

    Parameters
    ----------
    baselines: array
        array of baselines
    uncertainties: array
        uncertainties in gains for each antenna
    source_list: array
        source coordinates and intensity (l, m, B)
    """

    # propagate for a single visibility i.e. for a single baseline.
    # Should have errors in visiblities for each unique baseline

    # Loop over baselines
    for i in range(0, len(baselines)):
        pass
