import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from yaml import CLoader as Loader

import CRB
import power
import propagate
from misc import print_with_time


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

        if "telescope" in temp.keys():
            telescope = temp["telescope"]
        else:
            sys.exit("Please include telescope in config file")

        if "int_time" in temp.keys():
            int_time = float(temp["int_time"])
        else:
            sys.exit("Please include int_time in config file")

    return (
        ra,
        dec,
        T_sys,
        lamb,
        D,
        bandwidth,
        srclist,
        metafits,
        output,
        telescope,
        int_time,
    )


def get_source_list(filename, ra_ph, dec_ph, cut_off, lamb, D, output):
    """Returns a list of sources from a sky model

    Parameters
    ----------
    - filename: `string`
        filename of yaml sky model file
    - ra_ph: `float`
        ra of phase centre
    - dec_ph: `float`
        dec of phase centre
    - lamb: `float`
        wavelength
    - D: `float`
        distance between antenna

    Returns
    -------
    - source_list: `np.array`
        number of rows is number of sources
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


# Some notes about this program
# Initially sources are chosen by looking at sources in circle with diameter of the FOV
# centred around the phase centre.
# Assuming the phase centre is at zenith
def main():
    np.set_printoptions(linewidth=np.inf)

    if len(sys.argv) < 2:
        sys.exit("Please provide name of the config yaml file")

    config = sys.argv[1]
    (
        ra_ph,
        dec_ph,
        T_sys,
        lamb,
        D,
        bandwidth,
        srclist_dir,
        metafits_dir,
        output,
        telescope,
        int_time,
    ) = get_config(config)

    Path(output).mkdir(parents=True, exist_ok=True)

    print_with_time(
        f"INPUT SETTINGS: ra={ra_ph} dec={dec_ph} T_sys={T_sys} lambda={lamb} D={D}"
    )

    sigma = CRB.get_rms(T_sys, bandwidth, telescope, int_time)
    print_with_time(f"CALCULATED NOISE: {sigma}")
    cut_off = 5 * (sigma / np.sqrt(8256))
    print_with_time(f"CALCULATED CUT OFF FOR SOURCES: {cut_off}")

    # Get observations
    # get_obs_vec(metafits_dir)

    # Get source list
    print_with_time(f"READING IN SOURCE LIST FROM: {srclist_dir}")
    t1 = time.time()
    # NOTE: Should this be number of unique baselines
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
    if telescope == "mwa":
        l_arr, m_arr, beam = CRB.beam_form_mwa(0, 90, lamb, D, output)
    elif telescope == "ska":
        l_arr, m_arr, beam = CRB.beam_form_ska(0, 90, lamb, D, output)

    print_with_time("ATTENUATING WITH BEAM")
    sorted_source_list = CRB.attenuate(sorted_source_list, beam, l_arr, m_arr, output)
    print(sorted_source_list[-10:])

    # Calculate FIM
    print_with_time("CALCULATING THE FIM")
    uncertainties, baseline_lengths = CRB.calculate_fim(
        sorted_source_list, metafits_dir, lamb, sigma, output, telescope
    )

    mean_CRB = np.mean(uncertainties)

    # Propagate errors into visibilities
    print_with_time("PROPAGATING INTO VISIBILITIES")
    vis_uncertainties = propagate.propagate(
        baseline_lengths, sorted_source_list, uncertainties, lamb
    )

    print_with_time("BINNING ERRORS")
    vis_mat, u_arr, v_arr = power.uv_bin(
        lamb, vis_uncertainties, baseline_lengths, output
    )

    print_with_time("POWER SPECTRUM")
    power.power_bin(vis_mat, u_arr, v_arr, output)

    brightest = np.max(sorted_source_list)
    # Save pointing ra, pointing dec, mean CRB, num sources in FOV, brightest source in FOV
    with open("output_" + telescope + ".txt", "w") as f:
        f.write(
            f"{'ra':>15s} {'dec':>15s} {'CRB':>15s} {'num src':>15s} {'brightest':>15s}\n"
        )
        f.write(
            f"{ra_ph:15f} {dec_ph:15f} {mean_CRB:15.5e} {len(source_list):15d} {brightest:15.5f}\n"
        )

    # Save visibility uncertainties
    with open("vis_unc_" + telescope + ".txt", "w") as f:
        for row in vis_uncertainties:
            f.write(str(row) + "\n")


if __name__ == "__main__":
    main()
