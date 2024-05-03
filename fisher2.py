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

    deg_to_pi = np.pi/180.0
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
                if ( np.sqrt((dra * deg_to_pi)**2 + (drec * deg_to_pi)**2) > fov ): continue

                # Convert ra dec in deg to l, m direction cosines
                l = np.cos(dec) * np.sin(ra)
                m = np.sin(ra) * np.cos(dec_ph) - np.cos(dec) * \
                    np.sin(dec_ph) * np.cos(dra)

                if ('power_law' in data['flux_type']):
                    source_intensity = data['flux_type']['power_law']['fd']['i']

                if ('curved_power_law' in data['flux_type']):
                    source_intensity = data['flux_type']['curved_power_law']['fd']['i']

                if (source_intensity < cut_off): continue

                temp_array = [l, m, source_intensity]
                source_list.append(temp_array)

    return np.array(source_list)


@jit(nopython=True, cache=True, parallel=True)
#@jit(nopython=True, cache=True)
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

    fim_cos = np.zeros((num_ant, num_ant), dtype=np.complex64)

    num_sources = len(source_list)
    for k in range(0, num_sources):
        for l in prange(k + 1, num_sources):
            fim_cos[:, :] += 2 * source_list[k, 2] * source_list[l, 2] * (1 + np.cos(2 * np.pi * (baseline_lengths[:, :, 0] * (
                source_list[k, 0] - source_list[l, 0]) + baseline_lengths[:, :, 1] * (source_list[k, 1] - source_list[l, 1]))))

    fim_cos = 2.0/sigma**2 * fim_cos

    return fim_cos

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
        print(f'[{time_str}] CALCULATING TOOK: {t2 - t1}s')
        print(f'[{time_str}] IS THE FIM HERMITIAN?:  {ishermitian(fim_cos)}')

        with open(f'matrix_complex_cos_{obs}.txt', 'w') as testfile:
            for row in fim_cos:
                testfile.write(' '.join([str(a) for a in row]) + '\n')

        with open(f'matrix_abs_cos_{obs}.txt', 'w') as testfile:
            for row in fim_cos:
                testfile.write(' '.join([str(abs(a)) for a in row]) + '\n')

        # Calculate the CRB, which is the inverse of the FIM
        crb = np.linalg.inv(fim_cos)

        with open(f'crb_abs_{obs}.txt', 'w') as testfile:
            for row in crb:
                testfile.write(' '.join([str(abs(a)) for a in row]) + '\n')

        with open(f'crb_complex_{obs}.txt', 'w') as testfile:
            for row in crb:
                testfile.write(' '.join([str(a) for a in row]) + '\n')


        plt.matshow(abs(fim_cos))
        plt.colorbar()
        plt.title(obs + " FIM")
        plt.savefig("fim_cos.pdf")
        plt.clf()
        
        plt.matshow(abs(crb))
        plt.colorbar()
        plt.title(obs + " CRB")
        plt.savefig("crb.pdf")
        plt.clf()


# TODO: IMPLEMENT RADIOMETER EQUATION
if __name__ == '__main__':

    ra_ph = 72.5
    dec_ph = -13.35
    sigma = 200e-3

    srclist_dir = '/scratch/mwaeor/ejong/srclist/srclist_pumav3_EoR0LoBES_EoR1pietro_CenA-GP_2023-11-07.yaml'
    # srclist_dir = 'test.yaml'
    metafits_dir = '/scratch/mwaeor/ejong/SKAEOR15_145_data/rerun_1/solutions/'

    # Get observations
    get_obs_vec(metafits_dir)

    # Get source list
    time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f'[{time_str}] READING IN SOURCE LIST FROM: {srclist_dir}')
    t1 = time.time()
    source_list = get_source_list(srclist_dir, ra_ph, dec_ph, sigma, 2, 4.4)
    t2 = time.time()
    time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f'[{time_str}] READING IN TOOK: {t2 - t1}s')
    print(f'[{time_str}] NUMBER OF SOURCES: {len(source_list)}')

    # Calculate FIM
    print(f'[{time_str}] CALCULATING THE FIM')
    calculate_fim(source_list, metafits_dir, sigma)
