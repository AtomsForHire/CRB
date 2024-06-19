import datetime
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
import yaml
from mwa_qa import read_metafits
from numba import jit, prange
from scipy.linalg import ishermitian
from yaml import CLoader as Loader


def get_config(filename):
    with open(filename) as f:
        temp = yaml.safe_load(f)

        if "ra" in temp.keys():
            ra = temp["ra"]
        else:
            sys.exit("Please include ra in config file")

        if "dec" in temp.keys():
            dec = temp["dec"]
        else:
            sys.exit("Please include dec in config file")

        if "T_sys" in temp.keys():
            T_sys = float(temp["T_sys"])
        else:
            sys.exit("Please include T_sys in config file")

        if "lambda" in temp.keys():
            lamb = temp["lambda"]
        else:
            sys.exit("Please include lambda in config file")

        if "D" in temp.keys():
            D = temp["D"]
        else:
            sys.exit("Please include D in config file")

        if "srclist" in temp.keys():
            srclist = temp["srclist"]
        else:
            sys.exit("Please include srclist in config file")

        if "metafits" in temp.keys():
            metafits = temp["metafits"]
        else:
            sys.exit("Please include metafits in config file")

        if "bandwidth" in temp.keys():
            bandwidth = float(temp["bandwidth"])
        else:
            sys.exit("Please include bandwidth in config file")

        if "output" in temp.keys():
            output = temp["output"]
        else:
            sys.exit("Please include output in config file")

    return ra, dec, T_sys, lamb, D, bandwidth, srclist, metafits, output


def print_with_time(string):
    time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{time_str}] " + string)


def get_obs_vec(directory):
    """
    Get list of observation ids
    Input:
        - directory, string of path to directory containing metafits files
    Output:
        - order, ordered list of observation ids
    """
    order = list()
    for file in os.listdir(directory):
        if os.path.isfile(directory + file) and file.endswith(".metafits"):
            obsid = file.split(".")[0]
            order.append(obsid)

    order = sorted(order)

    return order


def get_source_list(filename, ra_ph, dec_ph, cut_off, lamb, D, output):
    """
    Returns a list of sources from a sky model

    Input:
        - filename, filename of yaml sky model file
        - ra_ph, ra of phase centre
        - dec_ph, dec of phase centre
        - lamb, wavelength
        - D, distance between antenna
    Output:
        - source_list, number of rows is number of sources
                       column 0 are l coords
                       column 1 are m coords
                       column 2 are source brightness
    """

    deg_to_rad = np.pi / 180.0
    fov = lamb / D
    with open(filename) as f:
        temp = yaml.load(f, Loader=Loader)

        num_sources = 0
        for key in temp:
            num_sources += len(temp[key])

        # store l, m, intensity values in this array
        # source_list = np.zeros((num_sources, 3))
        source_list = list()

        for key in temp:
            num_sources_in_key = len(temp[key])
            for i in range(0, num_sources_in_key):
                data = temp[key][i]
                ra = data["ra"]
                dec = data["dec"]

                dra = ra - ra_ph
                ddec = dec - dec_ph

                dist_from_ph = np.sqrt(
                    (dra * deg_to_rad) ** 2 + (ddec * deg_to_rad) ** 2
                )

                # Check if sources sit within FOV
                # if dist_from_ph > fov / 2.0:
                #     continue

                # Convert ra dec in deg to l, m direction cosines
                l = np.cos(dec) * np.sin(dra)
                m = np.sin(dec) * np.cos(dec_ph) - np.cos(dec) * np.sin(
                    dec_ph
                ) * np.cos(dra)

                if np.sqrt(l**2 + m**2) > np.sin(fov / 2.0):
                    continue

                if "power_law" in data["flux_type"]:
                    source_intensity = data["flux_type"]["power_law"]["fd"]["i"]

                if "curved_power_law" in data["flux_type"]:
                    source_intensity = data["flux_type"]["curved_power_law"]["fd"]["i"]

                if source_intensity < cut_off:
                    continue

                temp_array = [l, m, source_intensity]
                source_list.append(temp_array)

    if not source_list:
        print_with_time(
            f"NO SOURCES FOUND IN THIS FIELD WITH NOISE CALCULATED TO BE {cut_off}"
        )
        exit()

    circle1 = plt.Circle((0, 0), np.sin(fov / 2.0), color="r", alpha=0.2)
    temp = np.array(source_list)
    plt.scatter(temp[:, 0], temp[:, 1])
    ax = plt.gca()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.add_patch(circle1)
    plt.savefig(output + "/" + "sources.png")
    return np.array(source_list)


# @jit(nopython=True, cache=True, parallel=True)
@jit(nopython=True, cache=True)
def fim_loop(source_list, baseline_lengths, num_ant, sigma):
    """
    Function for executing the loop required to calculate the FIM
    separate from the wrapper to allow Numba to work

    Input:
        - source_list, 2D matrix of sources storing [l, m, intensity]
        - baseline_lengths, 2D matrix containing lengths between antenna
          i and j in both x and y coords [i, j, 0 or 1]
        - num_ant, number of antennas
    Output:
        - fim_cos, fisher information matrix
    """

    fim = np.zeros((num_ant, num_ant), dtype=np.complex64)

    num_sources = len(source_list)

    baseline_lengths = baseline_lengths / 2.0
    for a in range(0, num_ant):
        for b in range(a, num_ant):
            for i in range(0, num_sources):
                for j in range(0, num_sources):
                    fim[a, b] += (
                        source_list[i, 2]
                        * source_list[j, 2]
                        * np.exp(
                            -2
                            * np.pi
                            * 1j
                            * (
                                baseline_lengths[a, b, 0]
                                * (source_list[i, 0] - source_list[j, 0])
                                + baseline_lengths[a, b, 1]
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


def beam_form(az_point, alt_point, lamb, D, output):
    """Function for creating MWA beam

    Parameters
    ----------
    az_point: float
        azumith angle of the telescope's pointing (deg)
    alt_point: float
        altitude angle of the telescope's pointing (deg)
    lamb: float
        wavelength
    D: float
        length of tile
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

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    cp = ax.plot(x_loc, y_loc, ".", color="grey")

    ax.set_ylabel("Location Y metres")
    ax.set_xlabel("Location X metres")
    plt.savefig(output + "/" + "dipoles.png")

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


def calculate_fim(source_list, metafits_dir, sigma, output):
    """
    Input:
        - source_list, contains l, m and intensity values of sources
        - metafits_dir, string to directory containing <obsid>.metafits files
    """

    obsids = get_obs_vec(metafits_dir)

    for obs in obsids[0:1]:
        temp = read_metafits.Metafits(metafits_dir + str(obs) + ".metafits")

        # Calculate distances between each antenna
        baseline_lengths = np.zeros((temp.Nants, temp.Nants, 2))
        for i in range(0, temp.Nants):
            i_x, i_y, _ = temp.antenna_position_for(i)
            for j in range(0, temp.Nants):
                j_x, j_y, _ = temp.antenna_position_for(j)
                baseline_lengths[i, j, 0] = i_x - j_x
                baseline_lengths[i, j, 1] = i_y - j_y

        t1 = time.time()
        fim_cos = fim_loop(source_list, baseline_lengths, temp.Nants, sigma)
        t2 = time.time()

        time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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

        plt.matshow(abs(fim_cos))
        plt.colorbar()
        plt.title(obs + " FIM")
        plt.savefig(output + "/" + "fim_cos.pdf", bbox_inches="tight")
        plt.clf()

        plt.matshow(abs(crb))
        plt.colorbar()
        plt.title(obs + " CRB")
        plt.savefig(output + "/" + "crb.pdf", bbox_inches="tight")
        plt.clf()

        plt.plot(range(0, 128), diag)
        plt.savefig(output + "/" + "diag.pdf", bbox_inches="tight")


def find_lm_index(l, m, l_vec, m_vec):
    l_ind = np.argmin(abs(l - l_vec))
    m_ind = np.argmin(abs(m - m_vec))

    return l_ind, m_ind


def attenuate(source_list, beam, l_arr, m_arr, output):
    # l values are changing left to right in the matrix
    # So grab the first row for comparison
    l_vec = l_arr[0, :]

    # m values are changing in the columns
    m_vec = m_arr[:, 0]

    for i in range(0, len(source_list)):
        l_ind, m_ind = find_lm_index(source_list[i, 0], source_list[i, 1], l_vec, m_vec)
        # print(
        #     source_list[i, 0],
        #     source_list[i, 1],
        #     l_vec[l_ind],
        #     m_vec[m_ind],
        #     l_ind,
        #     m_ind,
        #     beam[l_ind, m_ind],
        # )
        source_list[i, 2] *= abs(beam[l_ind, m_ind])

    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    cp = ax.contourf(l_arr, m_arr, np.transpose(np.log10(abs(beam[:, :]))), 100)
    fig.colorbar(cp, ax=ax)  # Add a colorbar to a plot
    ax.scatter(source_list[:, 0], source_list[:, 1])
    ax.set_title("Autocorrelation beam")
    ax.set_xlabel("l")
    ax.set_ylabel("m")
    plt.savefig(output + "/" + "beam.png", bbox_inches="tight")

    return source_list


# https://slideplayer.com/slide/15019308/
def get_rms(T_sys, bandwidth):
    A_eff = 21
    t = 120

    return 10**26 * (2 * const.k * T_sys) / (A_eff * np.sqrt(bandwidth * t))


def main():

    if len(sys.argv) < 2:
        sys.exit("Please provide name of the config yaml file")

    config = sys.argv[1]
    ra_ph, dec_ph, T_sys, lamb, D, bandwidth, srclist_dir, metafits_dir, output = (
        get_config(config)
    )

    Path(output).mkdir(parents=True, exist_ok=True)

    print_with_time(
        f"INPUT SETTINGS: ra={ra_ph} dec={dec_ph} T_sys={T_sys} lambda={lamb} D={D}"
    )

    sigma = get_rms(T_sys, bandwidth)
    print_with_time(f"CALCULATED NOISE: {sigma}")
    cut_off = 5 * (sigma / np.sqrt(8256))
    print_with_time(f"CALCULATED CUT OFF FOR SOURCES: {cut_off}")

    # Get observations
    get_obs_vec(metafits_dir)

    # Get source list
    print_with_time(f"READING IN SOURCE LIST FROM: {srclist_dir}")
    t1 = time.time()
    # NOTE: Should this be number of unique baselines?
    # Or total number of baselines including redundant ones
    # which would surmount to a large number
    source_list = get_source_list(srclist_dir, ra_ph, dec_ph, cut_off, lamb, D, output)
    t2 = time.time()

    print_with_time(f"READING IN TOOK: {t2 - t1}s")
    print_with_time(f"NUMBER OF SOURCES: {len(source_list)}")
    print_with_time(f"TOP 10 SOURCES IN LIST ORDERED BY BRIGHTNESS")
    sorted_source_list = source_list[source_list[:, 2].argsort()]
    print(sorted_source_list[-10:])

    print_with_time("CALCULATING BEAM")
    l_arr, m_arr, beam = beam_form(0, 90, lamb, D, output)

    print_with_time("ATTENUATING WITH BEAM")
    sorted_source_list = attenuate(sorted_source_list, beam, l_arr, m_arr, output)
    print(sorted_source_list[-10:])

    # Calculate FIM
    print_with_time("CALCULATING THE FIM")
    calculate_fim(sorted_source_list, metafits_dir, sigma, output)


if __name__ == "__main__":
    main()
