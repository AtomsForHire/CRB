import datetime
import itertools
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
from mwa_qa import read_metafits
from numba import jit, prange
from scipy.linalg import ishermitian
from scipy.special import jv

from misc import print_with_time

# BUG: For some reason if I turn off parallel mode in this function
# it causes an out of memory error (OOM or seg fault) while trying
# to bin errors later on in propagate.propagate. But if I turn off
# parallel jit in this function here, and turn off jit all together
# in propagate.propagate then it works fine.


# @jit(nopython=True, cache=True)
@jit(nopython=True, cache=True, parallel=True)
def fim_loop(source_list, baseline_lengths, num_ant, lamb, sigma, chan_freq):
    """Function for executing the loop required to calculate the FIM
    separate from the wrapper to allow Numba to work

    Parameters
    ----------
    - source_list: `np.array`
        2D matrix of sources storing [l, m, intensity, spectral_index, freq]
    - baseline_lengths: `np.array`
        2D matrix containing lengths between antenna
        i and j in both x and y coords [i, j, 0 or 1]
    - num_ant: `int`
        number of antennas
    - lamb: `float`
        observing wavelength
    - sigma: `float`
        value from radiometer equation
    - chan_freq: `float`
        frequency of current channel

    Returns
    -------
    fim_cos: `np.array`
        fisher information matrix
    """

    fim = np.zeros((num_ant, num_ant), dtype=np.complex64)

    num_sources = len(source_list)

    original_intensities = source_list[:, 2]
    spectral_indices = source_list[:, 3]
    freqs = source_list[:, 4]

    source_intensities = (
        original_intensities * (chan_freq / freqs[:]) ** spectral_indices
    )

    # Make baselines in wavelengths
    # baseline_lengths = baseline_lengths / 2.0
    baselines = baseline_lengths / lamb
    for a in prange(0, num_ant):
        for b in range(a, num_ant):
            for i in range(0, num_sources):
                for j in range(0, num_sources):
                    fim[a, b] += (
                        source_intensities[i]
                        * source_intensities[j]
                        * np.exp(
                            -2
                            * np.pi
                            * 1j
                            * (
                                baselines[a, b, 0]
                                * (source_list[i, 0] - source_list[j, 0])
                                + baselines[a, b, 1]
                                * (source_list[i, 1] - source_list[j, 1])
                            )
                        )
                    )

                    if a == b:
                        fim[a, b] += 127 * (
                            source_list[i, 2] * source_list[j, 2]
                        ) + 4 * (source_list[i, 2] * source_list[j, 2])

    for a in range(0, num_ant):
        for b in range(a, num_ant):
            fim[b, a] = np.conjugate(fim[a, b])

    fim = 2.0 / sigma**2 * fim

    return fim


# TODO: Implement some beam-steering thing
def beam_form_ska(az_point, alt_point, lamb, D, output):
    """Function for creating SKA beam

    Parameters
    ----------
    - az_point: `float`
        azumith angle of the telescope's pointing (deg)
    - alt_point: `float`
        altitude angle of the telescope's pointing (deg)
    - lamb: `float`
        wavelength
    - D: `float`
        length of tile
    - output: `string`
        output path

    Returns
    -------
    - l_arr: `np.array`
        array of l values
    - m_arr: `np.array`
        array of m values
    - beam: `np.array`
        array of beam
    """

    deg_to_pi = np.pi / 180.0
    az_point = az_point * deg_to_pi
    alt_point = alt_point * deg_to_pi

    # Convert alt and az to l and m
    l_point = np.sin(np.pi / 2.0 - alt_point) * np.sin(az_point)
    m_point = np.sin(np.pi / 2.0 - alt_point) * np.cos(az_point)

    # NOTE: I don't know what this variable is
    N_ortho = 1024

    # NOTE: Unlike the MWA beam, we can approximate the SKA beam with a airy disk function
    lmax = 1.0
    l = (np.arange(N_ortho) - N_ortho / 2.0) / (N_ortho / 2.0) * lmax
    m = l

    l_arr, m_arr = np.meshgrid(l, m)

    # Max intensity at centre
    I0 = 1
    # Radius of aperture
    a = D

    # Will need to convert coordinates (l,m) into some theta from the centre of the beam
    beam = np.zeros((N_ortho, N_ortho))
    k = 2 * np.pi / lamb

    R = np.sqrt(l_arr**2 + m_arr**2)
    x = k * a * R / 2
    beam = (2 * jv(1, x) / x) ** 2
    beam[R == 0] = 1

    # save full beam
    # fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    # cp = ax.contourf(l_arr, m_arr, beam[:, :], 100)
    # fig.colorbar(cp, ax=ax)  # Add a colorbar to a plot
    # ax.set_title("Airy Disk")
    # ax.set_xlabel("l")
    # ax.set_ylabel("m")
    # plt.savefig(output + "/" + "full_beam.png", bbox_inches="tight")

    # Create mask, anything outside the fov set to 0
    fov = lamb / D

    l_fov = np.sin(fov / 2.0)
    mask = np.zeros((N_ortho, N_ortho), dtype="float")
    for i in range(N_ortho):
        for j in range(N_ortho):
            if np.sqrt(l[i] ** 2 + m[j] ** 2) <= l_fov:
                mask[i, j] = 1.0

    beam *= mask

    # save masked beam
    # fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    # cp = ax.contourf(l_arr, m_arr, beam[:, :], 100)
    # fig.colorbar(cp, ax=ax)  # Add a colorbar to a plot
    # ax.set_title("Airy Disk")
    # ax.set_xlabel("l")
    # ax.set_ylabel("m")
    # plt.savefig(output + "/" + "full_beam_masked.png", bbox_inches="tight")

    return l_arr, m_arr, beam


def beam_form_mwa(az_point, alt_point, lamb, D, output):
    """Function for creating MWA beam

    Parameters
    ----------
    - az_point: `float`
        azumith angle of the telescope's pointing (deg)
    - alt_point: `float`
        altitude angle of the telescope's pointing (deg)
    - lamb: `float`
        wavelength
    - D: `float`
        length of tile
    - output: `string`
        output path

    Returns
    -------
    - l_arr: `np.array`
        array of l values
    - m_arr: `np.array`
        array of m values
    - beam: `np.array`
        array of beam
    """

    deg_to_pi = np.pi / 180.0
    az_point = az_point * deg_to_pi
    alt_point = alt_point * deg_to_pi

    # Convert alt and az to l and m
    l_point = np.sin(np.pi / 2.0 - alt_point) * np.sin(az_point)
    m_point = np.sin(np.pi / 2.0 - alt_point) * np.cos(az_point)

    # NOTE: I don't know what this variable is
    N_ortho = 1024

    # Create dipoles in correct positions
    sep = 1.1
    x_len = 4
    y_len = x_len
    n_ant = x_len * y_len

    x_loc = np.zeros(n_ant)
    y_loc = np.zeros(n_ant)

    count = 0
    for i in range(0, x_len):
        for j in range(0, y_len):
            x_loc[count] = i * sep
            y_loc[count] = j * sep
            count += 1

    # Shift everything to be centred around the centre of the tile
    cent = [np.mean(x_loc), np.mean(y_loc)]

    x_loc -= cent[0]
    y_loc -= cent[1]

    # fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # cp = ax.plot(x_loc, y_loc, ".", color="grey")

    # ax.set_ylabel("Location Y metres")
    # ax.set_xlabel("Location X metres")
    # plt.savefig(output + "/" + "dipoles.png")

    # Create beams
    station_beam_Xtheta = np.zeros((N_ortho, N_ortho), dtype="complex")
    station_beam_Ytheta = np.zeros((N_ortho, N_ortho), dtype="complex")
    station_beam_Xphi = np.zeros((N_ortho, N_ortho), dtype="complex")
    station_beam_Yphi = np.zeros((N_ortho, N_ortho), dtype="complex")

    # Set up l,m grid
    lmax = 1.0
    l = (np.arange(N_ortho) - N_ortho / 2.0) / (N_ortho / 2.0) * lmax
    m = l

    # NOTE: Don't know what this does
    tot_phase = np.zeros((N_ortho, N_ortho), dtype="float")

    # Create mask, anything outside the fov set to 0
    fov = lamb / D

    l_fov = np.sin(fov / 2.0)
    mask = np.zeros((N_ortho, N_ortho), dtype="float")
    for i in range(N_ortho):
        for j in range(N_ortho):
            if np.sqrt(l[i] ** 2 + m[j] ** 2) <= l_fov:
                mask[i, j] = 1.0

    # NOTE: I don't know what's going on here
    for k in range(n_ant):
        phase_x = (x_loc[k]) / lamb
        phase_y = (y_loc[k]) / lamb

        for j in range(N_ortho):
            tot_phase[:, j] = (phase_x * (l - l_point)) + (phase_y * (m[j] - m_point))

        station_beam_Xtheta[:, :] += (
            np.cos(2.0 * np.pi * tot_phase) - 1j * np.sin(2.0 * np.pi * tot_phase)
        ) * mask
        station_beam_Ytheta[:, :] += (
            np.cos(2.0 * np.pi * tot_phase) - 1j * np.sin(2.0 * np.pi * tot_phase)
        ) * mask

    XX = np.zeros((N_ortho, N_ortho), dtype="complex")
    YY = XX
    XY = XX
    YX = XX

    XX = station_beam_Xtheta[:, :] * np.conj(
        station_beam_Xtheta[:, :]
    ) + station_beam_Xphi[:, :] * np.conj(station_beam_Xphi[:, :])
    YY = station_beam_Ytheta[:, :] * np.conj(
        station_beam_Ytheta[:, :]
    ) + station_beam_Yphi[:, :] * np.conj(station_beam_Yphi[:, :])
    XY = station_beam_Xtheta[:, :] * np.conj(
        station_beam_Ytheta[:, :]
    ) + station_beam_Xphi[:, :] * np.conj(station_beam_Yphi[:, :])
    YX = station_beam_Ytheta[:, :] * np.conj(
        station_beam_Xtheta[:, :]
    ) + station_beam_Yphi[:, :] * np.conj(station_beam_Xphi[:, :])

    # Normalise
    XX = XX / np.max(XX)
    YY = YY / np.max(YY)

    stokes_I = 0.5 * (XX + YY)
    # Save beam
    l_arr, m_arr = np.meshgrid(l, m)

    # fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    # cp = ax.contourf(l_arr, m_arr, np.transpose(np.log10(abs(stokes_I[:, :]))), 100)
    # fig.colorbar(cp, ax=ax)  # Add a colorbar to a plot
    # ax.set_title("Autocorrelation beam")
    # ax.set_xlabel("l")
    # ax.set_ylabel("m")
    # plt.savefig(output + "/" + "beam.png", bbox_inches="tight")

    return l_arr, m_arr, stokes_I


def create_MWA_baselines(metafits_path):
    """Function for extracting MWA baselines from MWA metafits file

    Parameters
    ----------
    metafits_dir: `string`
        path to MWA metafits file
    obs: `int`
        observation id

    Returns
    -------
    baseline_lengths: `np.array`
        matrix of baselines between antennas i and j: [i, j, (x,y)]
    """
    temp = read_metafits.Metafits(metafits_path)
    baseline_lengths = np.zeros((temp.Nants, temp.Nants, 2))
    for i in range(0, temp.Nants):
        i_x, i_y, _ = temp.antenna_position_for(i)
        for j in range(0, temp.Nants):
            j_x, j_y, _ = temp.antenna_position_for(j)
            baseline_lengths[i, j, 0] = i_x - j_x
            baseline_lengths[i, j, 1] = i_y - j_y

    return baseline_lengths


def calculate_fim(
    source_list, baseline_lengths, num_ant, lamb, chan_freq, sigma, output
):
    """Wrapper function for calculating the FIM

    Parameters
    ----------
    - source_list: `np.array`
        contains l, m and intensity values of sources
    - baseline_lengths: `np.array`
        2D array of baselines
    - num_ant: `int`
        number of antennas
    - lamb: `float`
        observing wavelength
    - chan_freq: `float`
        observing frequency
    - sigma: `float`
        value from radiometer equation
    - output: `string`
        place to save output

    Returns
    -------
    - result: `float`
        mean value of the diagonal of the square-rooted CRB matrix
    """

    t1 = time.time()
    fim_cos = fim_loop(source_list, baseline_lengths, num_ant, lamb, sigma, chan_freq)
    t2 = time.time()

    print_with_time(f"CALCULATING TOOK: {t2 - t1}s")
    print_with_time(f"IS THE FIM HERMITIAN?:  {ishermitian(fim_cos)}")

    with open(output + "/" + "matrix_complex.txt", "w") as f:
        for row in fim_cos:
            f.write(" ".join([str(a) for a in row]) + "\n")

    with open(output + "/" + "matrix_abs.txt", "w") as f:
        for row in fim_cos:
            f.write(" ".join([str(abs(a)) for a in row]) + "\n")

    # Calculate the CRB, which is the inverse of the FIM
    crb = np.sqrt(np.linalg.inv(fim_cos))

    with open(output + "/" + "crb_abs.txt", "w") as f:
        for row in crb:
            f.write(" ".join([str(abs(a)) for a in row]) + "\n")

    with open(output + "/" + "crb_complex.txt", "w") as f:
        for row in crb:
            f.write(" ".join([str(a) for a in row]) + "\n")

    diag = np.diagonal(abs(crb))

    with open(output + "/" + "diag.txt", "w") as f:
        for row in diag:
            f.write(str(row) + "\n")

        # plt.matshow(abs(fim_cos))
        # plt.colorbar()
        # plt.title(obs + " FIM")
        # plt.savefig(output + "/" + "fim_cos.pdf", bbox_inches="tight")
        # plt.clf()

        # plt.matshow(abs(crb))
        # plt.colorbar()
        # plt.title(obs + " CRB")
        # plt.savefig(output + "/" + "crb.pdf", bbox_inches="tight")
        # plt.clf()

        # plt.plot(range(0, 128), diag)
        # plt.savefig(output + "/" + "diag.pdf", bbox_inches="tight")

        return diag


@jit(nopython=True, cache=True)
def find_lm_index(l, m, l_vec, m_vec):
    l_ind = np.argmin(np.abs(l - l_vec))
    m_ind = np.argmin(np.abs(m - m_vec))

    return l_ind, m_ind


@jit(nopython=True, cache=True)
def attenuate(source_list, beam, l_arr, m_arr, output):
    """Function for applying beam attenuation onto sources within FOV

    Parameters
    ----------
    source_list: `np.array`
        array of sources containing [l, m, B] values
    beam: `np.array`
        array of beam sensitivity values
    l_arr: `np.array`
        array of l values for beam
    m_arr: `np.array`
        array of m values for beam
    output: string
        path to directory to save stuff

    Returns
    -------
    source_list: `np.array`
        source_list after attenuation
    """
    # l values are changing left to right in the matrix
    # So grab the first row for comparison
    l_vec = l_arr[0, :]

    # m values are changing in the columns
    m_vec = m_arr[:, 0]

    for i in range(0, len(source_list)):
        l_ind, m_ind = find_lm_index(source_list[i, 0], source_list[i, 1], l_vec, m_vec)
        source_list[i, 2] *= abs(beam[l_ind, m_ind])

    # fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    # cp = ax.contourf(l_arr, m_arr, np.transpose(np.log10(abs(beam[:, :]))), 100)
    # fig.colorbar(cp, ax=ax)  # Add a colorbar to a plot
    # ax.scatter(source_list[:, 0], source_list[:, 1], s=0.5, alpha=0.5)
    # ax.set_title("Autocorrelation beam")
    # ax.set_xlabel("l")
    # ax.set_ylabel("m")
    # plt.savefig(output + "/" + "masked_beam_with_sources.png", bbox_inches="tight")

    return source_list


# https://slideplayer.com/slide/15019308/
def get_rms(T_sys, bandwidth, telescope, int_time):
    """Calculate the radio interferometer equation

    Parameters
    ----------
    - T_sys: `float`
        system noise
    - bandwidth: `float`
        bandwidth of a measurement
    - telescope: `string`
        mwa or ska
    - int_time: `float`
        integration time

    Returns
    -------
    result: `float`
        result of equation
    """

    if telescope == "mwa":
        A_eff = 21
    elif telescope == "ska":
        A_eff = np.pi * (35 / 2) ** 2

    return 10**26 * (2 * const.k * T_sys) / (A_eff * np.sqrt(bandwidth * int_time))
