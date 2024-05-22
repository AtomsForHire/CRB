import numpy as np
import matplotlib.pyplot as plt
import yaml
from scipy.linalg import ishermitian
from mwa_qa import read_metafits
from numba import jit, prange
import sys
import os
import time
import datetime
import scipy.constants as const

def get_config(filename):
    with open(filename) as f:
        temp = yaml.safe_load(f)

        if ("ra" in temp.keys()):
            ra = temp["ra"]
        else:
            sys.exit("Please include ra in config file")

        if ("dec" in temp.keys()):
            dec = temp["dec"]
        else:
            sys.exit("Please include dec in config file")

        if ("T_sys" in temp.keys()):
            T_sys = float(temp["T_sys"])
        else:
            sys.exit("Please include T_sys in config file")

        if ("lambda" in temp.keys()):
            lamb = temp["lambda"]
        else:
            sys.exit("Please include lambda in config file")

        if ("D" in temp.keys()):
            D = temp["D"]
        else:
            sys.exit("Please include D in config file")

        if ("srclist" in temp.keys()):
            srclist = temp["srclist"]
        else:
            sys.exit("Please include srclist in config file")

        if ("metafits" in temp.keys()):
            metafits = temp["metafits"]
        else:
            sys.exit("Please include metafits in config file")

    return ra, dec, T_sys, lamb, D, srclist, metafits


def print_with_time(string):
    time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f'[{time_str}] ' + string)


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
            obsid = file.split('.')[0]
            order.append(obsid)

    order = sorted(order)

    return order

def get_source_list(filename, ra_ph, dec_ph, cut_off, lamb, D):
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

    deg_to_rad = np.pi/180.0
    fov = lamb/D
    with open(srclist_dir) as f:
        temp = yaml.safe_load(f)

        num_sources = 0
        for key in temp:
            num_sources += len(temp[key])

        # store l, m, intensity values in this array
        # source_list = np.zeros((num_sources, 3))
        source_list = list()

        # continue_from = 0
        for key in temp:
            num_sources_in_key = len(temp[key])
            for i in range(0, num_sources_in_key):
                data = temp[key][i]
                ra = data['ra']
                dec = data['dec']

                dra = ra - ra_ph
                drec = dec - dec_ph

                # Check if sources sit within FOV
                if ( np.sqrt((dra * deg_to_rad)**2 + (drec * deg_to_rad)**2) > fov/2.0 ): continue

                # Convert ra dec in deg to l, m direction cosines
                l = np.cos(dec) * np.sin(ra)
                m = np.sin(ra) * np.cos(dec_ph) - np.cos(dec) * \
                    np.sin(dec_ph) * np.cos(dra)

                if ('power_law' in data['flux_type']):
                    source_intensity = data['flux_type']['power_law']['fd']['i']

                if ('curved_power_law' in data['flux_type']):
                    source_intensity = data['flux_type']['curved_power_law']['fd']['i']

                if (source_intensity < cut_off): continue

                temp_array = [ra, dec, source_intensity]
                source_list.append(temp_array)

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
    # for k in range(0, num_sources):
    #     for l in prange(k + 1, num_sources):
    #         fim_cos[:, :] += 2 * source_list[k, 2] * source_list[l, 2] * (1 + np.cos(2 * np.pi * (baseline_lengths[:, :, 0] * (
    #             source_list[k, 0] - source_list[l, 0]) + baseline_lengths[:, :, 1] * (source_list[k, 1] - source_list[l, 1]))))

    baseline_lengths = baseline_lengths / 2.0
    for a in range(0, num_ant):
        for b in range(a, num_ant):
            for i in range(0, num_sources):
                for j in range(0, num_sources):
                    fim[a, b] += source_list[i, 2] * source_list[j, 2] * np.exp(-2 * np.pi * 1j *
                                                                               (baseline_lengths[a, b, 0] * (source_list[i, 0] - source_list[j, 0]) +
                                                                                baseline_lengths[a, b, 1] * (source_list[i, 1] - source_list[j, 1])))

                    if ( a == b ):
                        fim[a, b] += 127 * (source_list[i, 2] * source_list[j, 2]) + 4 * (source_list[i, 2] * source_list[j, 2])

    for a in range(0, num_ant):
        for b in range(a, num_ant):
            fim[b, a] = np.conjugate(fim[a, b])

    # for i in range(0, num_sources):
    #     for j in range(0, num_sources):
    #         fim[:, :] += source_list[i, 2] * source_list[j, 2] * np.exp(-2 * np.pi * 1j *
    #                                                                    (baseline_lengths[:, :, 0] * (source_list[i, 0] - source_list[j, 0]) +
    #                                                                     baseline_lengths[:, :, 1] * (source_list[i, 1] - source_list[j, 1])))


    fim = 2.0/sigma**2 * fim

    return fim


def beam_form():
    pass

def calculate_fim(source_list, metafits_dir, sigma):
    """
        Input:
            - source_list, contains l, m and intensity values of sources
            - metafits_dir, string to directory containing <obsid>.metafits files
    """

    obsids = get_obs_vec(metafits_dir)

    for obs in obsids[0:1]:
        temp = read_metafits.Metafits(metafits_dir + str(obs) + '.metafits')

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
        print_with_time(f'CALCULATING TOOK: {t2 - t1}s')
        print_with_time(f'IS THE FIM HERMITIAN?:  {ishermitian(fim_cos)}')

        with open(f'matrix_complex.txt', 'w') as f:
            for row in fim_cos:
                f.write(' '.join([str(a) for a in row]) + '\n')

        with open(f'matrix_abs.txt', 'w') as f:
            for row in fim_cos:
                f.write(' '.join([str(abs(a)) for a in row]) + '\n')

        # Calculate the CRB, which is the inverse of the FIM
        crb = np.sqrt(np.linalg.inv(fim_cos))

        with open(f'crb_abs.txt', 'w') as f:
            for row in crb:
                f.write(' '.join([str(abs(a)) for a in row]) + '\n')

        with open(f'crb_complex.txt', 'w') as f:
            for row in crb:
                f.write(' '.join([str(a) for a in row]) + '\n')

        diag = np.diagonal(abs(crb))

        with open(f'diag.txt', 'w') as f:
            for row in diag:
                f.write(str(row) + '\n')

        plt.matshow(abs(fim_cos))
        plt.colorbar()
        plt.title(obs + " FIM")
        plt.savefig("fim_cos.pdf", bbox_inches = "tight")
        plt.clf()
        
        plt.matshow(abs(crb))
        plt.colorbar()
        plt.title(obs + " CRB")
        plt.savefig("crb.pdf", bbox_inches = "tight")
        plt.clf()
        
        plt.plot(range(0, 128), diag)
        plt.savefig("diag.pdf", bbox_inches = "tight")


# https://slideplayer.com/slide/15019308/
def get_rms(T_sys):
    A_eff = 21
    bandwidth = 10e3
    t = 120

    return 10**(-26) * (2 * const.k * T_sys) / (A_eff * np.sqrt(bandwidth * t))


if __name__ == '__main__':

    if (len(sys.argv) < 2):
        sys.exit("Please provide name of the config yaml file")

    config = sys.argv[1]
    ra_ph, dec_ph, T_sys, lamb, D, srclist_dir, metafits_dir = get_config(config)
    print_with_time(f'INPUT SETTINGS: ra={ra_ph} dec={dec_ph} T_sys={T_sys} lambda={lamb} D={D}')

    sigma = get_rms(T_sys)
    print_with_time(f'CALCULATED NOISE: {sigma}')
    sys.exit()

    # Get observations
    get_obs_vec(metafits_dir)

    # Get source list
    print_with_time(f'READING IN SOURCE LIST FROM: {srclist_dir}')
    t1 = time.time()
    source_list = get_source_list(srclist_dir, ra_ph, dec_ph, sigma, 2, 4.4)
    t2 = time.time()

    print_with_time(f'READING IN TOOK: {t2 - t1}s')
    print_with_time(f'NUMBER OF SOURCES: {len(source_list)}')
    print_with_time(f'TOP 10 SOURCES IN LIST ORDERED BY BRIGHTNESS')
    sorted_source_list = source_list[source_list[:, 2].argsort()]
    print(sorted_source_list[-10:])


    # Calculate FIM
    print_with_time(f'CALCULATING THE FIM')
    calculate_fim(source_list, metafits_dir, sigma)
