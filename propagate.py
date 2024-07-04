import numpy as np
from numba import jit

from misc import print_with_time


@jit(nopython=True, cache=True)
def partial(u, v, gain, source_list):
    """Function for evaluating the partial derivative expression

    Parameters
    ----------
    - u: `float`
        baseline coord
    - v: `float`
        baseline coord
    - source_list: `array`
        source coordinates and intensity (l, m, B)
    - gain: `float`
        the gain of the antenna

    Returns
    -------
    - result: `complex`
        result of the partial derivative expression
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
    - u: `float`
        baseline coord
    - v: `float`
        baseline coord
    - source_list: `array`
        source coordinates and intensity (l, m, B)
    - gain: `float`
        the gain of the antenna

    Returns
    -------
    - result: `complex`
        result of the partial derivative expression
    """
    result = 0
    for i in range(0, len(source_list)):
        result += source_list[i, 2] * np.exp(
            2.0 * np.pi * 1j * (u * source_list[i, 0] + v * source_list[i, 1])
        )

    return result * gain


@jit(nopython=True, cache=True)
def propagate(baseline_lengths, source_list, uncertainties, lamb):
    """Function for propagating gain errors into visibilities

    Parameters
    ----------
    - baselines: `array`
        array of baselines
    - uncertainties: `array`
        uncertainties in gains for each antenna
    - source_list: `array`
        source coordinates and intensity (l, m, B)

    Returns
    -------
    - vis_uncertainties: `np.array`
        matrix of uncertainties for baselines form by antennas i and j
    """

    # Propagate for a single visibility, i.e. a single baseline

    num_ant = baseline_lengths.shape[0]
    vis_uncertainties = np.zeros((num_ant, num_ant), dtype="float")

    baselines = baseline_lengths / lamb
    for a in range(0, num_ant):
        g1_unc = uncertainties[a]
        g1 = 1
        for b in range(0, num_ant):
            g2_unc = uncertainties[b]
            g2 = 1
            u = baselines[a, b, 0]
            v = baselines[a, b, 1]

            dvdg1 = partial(u, v, g1, source_list)
            dvdg2 = partial(u, v, g2, source_list)
            vis_uncertainties[a, b] = (
                np.abs(dvdg1) ** 2 * g1_unc**2 + np.abs(dvdg2) ** 2 * g2_unc**2
            )

    return np.sqrt(vis_uncertainties)
