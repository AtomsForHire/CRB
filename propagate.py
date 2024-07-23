import numpy as np
from numba import jit, prange


@jit(nopython=True, cache=True, parallel=True)
def realise(vis_uncertainties):
    """Function to realize the visibility errors by sampling 0-mean vis

    Parameters
    ----------
    - vis_uncertainties: `array`
        2D array of visibility errors

    Returns
    -------
    - realised_unc: `array`
    """

    num_ant = vis_uncertainties.shape[0]

    realised_unc = np.zeros((num_ant, num_ant), dtype=np.complex64)
    for a in prange(0, num_ant):
        for b in range(0, num_ant):
            if vis_uncertainties[a, b].imag > 1e-30:
                print(a, b, vis_uncertainties[a, b].imag)
            sigma = vis_uncertainties[a, b].real / np.sqrt(2)
            real = np.random.normal(0, sigma)
            imag = np.random.normal(0, sigma)

            realised_unc[a, b] = real + 1j * imag

    return realised_unc


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
    for i in prange(0, len(source_list)):
        result += source_list[i, 2] * np.exp(
            -2.0 * np.pi * 1j * (u * source_list[i, 0] + v * source_list[i, 1])
        )

    return result * gain


def partial_star(u, v, gain, source_list):
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
    for i in prange(0, len(source_list)):
        result += source_list[i, 2] * np.exp(
            2.0 * np.pi * 1j * (u * source_list[i, 0] + v * source_list[i, 1])
        )

    return result * gain


# @jit(nopython=True, cache=True, parallel=True)
def propagate(baseline_lengths, source_list, covar_mat, lamb):
    """Function for propagating gain errors into visibilities

    Parameters
    ----------
    - baselines: `array`
        array of baselines
    - source_list: `array`
        source coordinates and intensity (l, m, B)
    - covar_mat: `array`
        should be the whole calculated CRB matrix (not square-rooted).

    Returns
    -------
    - vis_uncertainties: `np.array`
        matrix of uncertainties for baselines form by antennas i and j
    """

    # Propagate for a single visibility, i.e. a single baseline

    num_ant = baseline_lengths.shape[0]
    vis_uncertainties = np.zeros((num_ant, num_ant), dtype=np.complex64)

    baselines = baseline_lengths / lamb
    # Loop through antennas
    for a in prange(0, num_ant):
        g1_var = covar_mat[a, a]
        g1 = 1
        for b in range(0, num_ant):
            g2_var = covar_mat[b, b]

            g1g2_covar = covar_mat[a, b]
            g2g1_covar = covar_mat[b, a]

            g2 = 1
            u = baselines[a, b, 0]
            v = baselines[a, b, 1]

            dvdg1 = partial(u, v, g1, source_list)
            dvdg2 = partial(u, v, g2, source_list)
            dvdg1_star = partial_star(u, v, g1, source_list)
            dvdg2_star = partial_star(u, v, g2, source_list)
            vis_uncertainties[a, b] = np.abs(dvdg1) ** 2 * g1_var**2
            +np.abs(dvdg2) ** 2 * g2_var**2
            +dvdg1_star * dvdg2 * g2g1_covar
            +dvdg2_star * dvdg1 * g1g2_covar

    return np.sqrt(vis_uncertainties)
