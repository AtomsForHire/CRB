import numpy as np
import yaml
from numba import jit
from yaml import CLoader as Loader

from misc import print_with_time


def get_source_list(filename, ra_ph, dec_ph, cut_off, output):
    """Returns a list of sources from a sky model, sources are chosen if they are above the threshold

    Parameters
    ----------
    - filename: `string`
        filename of yaml sky model file
    - ra_ph: `float`
        ra of phase centre
    - dec_ph: `float`
        dec of phase centre

    Returns
    -------
    - source_list: `np.array`
        number of rows is number of sources
        column 0 are l coords
        column 1 are m coords
        column 2 are source brightness
    """

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

                # dist_from_ph = np.sqrt(
                #     (dra * deg_to_rad) ** 2 + (ddec * deg_to_rad) ** 2
                # )

                # Check if sources sit within FOV
                # if dist_from_ph > fov / 2.0:
                #     continue

                # Convert ra dec in deg to l, m direction cosines
                l = np.cos(dec) * np.sin(dra)
                m = np.sin(dec) * np.cos(dec_ph) - np.cos(dec) * np.sin(
                    dec_ph
                ) * np.cos(dra)

                # if np.sqrt(l**2 + m**2) > np.sin(fov / 2.0):
                #     continue

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

    # circle1 = plt.Circle((0, 0), np.sin(fov / 2.0), color="r", alpha=0.2)
    # temp = np.array(source_list)
    # plt.scatter(temp[:, 0], temp[:, 1])
    # ax = plt.gca()
    # ax.set_xlim(-1, 1)
    # ax.set_ylim(-1, 1)
    # ax.add_patch(circle1)
    # plt.savefig(output + "/" + "sources.png")
    return np.array(source_list)


@jit
def fov_cut(source_list, lamb, D):
    """Function to select sources within an FOV defined by the observing frequency

    Parameters
    ----------
    - source_list: `np.array`
        array containing all the sources above threshold
    - lamb: `float`
        observing frequency
    - D: `float`
        area of tile

    Returns
    -------
    - fov_cut_source_list: `np.array`
        array of sources inside FOV
    """
    # deg_to_rad = np.pi / 180.0
    fov = lamb / D

    # if np.sqrt(l**2 + m**2) > np.sin(fov / 2.0):
    #     continue

    l_arr = source_list[:, 0]
    m_arr = source_list[:, 1]
    dist = np.sqrt(l_arr**2 + m_arr**2)

    return source_list[dist < np.sin(fov / 2.0), :]
