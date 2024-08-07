import random
import sys
import time
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
import yaml
from yaml import CLoader as Loader

import CRB
import power
import propagate
import sources
from misc import print_with_time


def get_config(filename):
    with open(filename) as f:
        temp = yaml.load(f, Loader=Loader)

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

        if "start_freq" in temp.keys():
            start_freq = float(temp["start_freq"])
        else:
            sys.exit("Please include start_freq in config file")

        if "end_freq" in temp.keys():
            end_freq = float(temp["end_freq"])
        else:
            sys.exit("Please include end_freq in config file")

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

        if "channel_width" in temp.keys():
            channel_width = float(temp["channel_width"])
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

        if "obs_name" in temp.keys():
            obs_name = temp["obs_name"]
        else:
            sys.exit("Please include obs_name in config file")

    return (
        ra,
        dec,
        T_sys,
        start_freq,
        end_freq,
        D,
        channel_width,
        srclist,
        metafits,
        output,
        telescope,
        int_time,
        obs_name,
    )


def save_hdf5(
    k_perp,
    power_spec,
    output,
    telescope,
    channel_width,
    start_freq,
    end_freq,
    num_freq,
    ra,
    dec,
    name,
):
    """Function to save complex matrix to power spectrum

    Parameters
    ----------
    - x_axis: `np.array`
        Should be array of k_perp
    - y_axis: `np.array`
        Should be array of k_parallel
    - power_space: `np.array`
        matrix to save
    - output: `string`
        path to directory for saving
    - telescope: `string`
        telescope

    Returns
    -------
    None
    """
    with h5py.File(output + "/output_" + telescope + ".hdf5", "w") as f:
        f.attrs["obs_name"] = name
        f.attrs["ra"] = ra
        f.attrs["dec"] = dec
        f.attrs["telescope"] = telescope
        f.attrs["chan_width"] = channel_width
        f.attrs["start_freq"] = start_freq
        f.attrs["end_freq"] = end_freq
        f.attrs["num_freq"] = num_freq
        f.create_dataset("k_perp", data=k_perp)
        f.create_dataset("power_spec", data=power_spec)


# NOTE: Some notes about this program
# Initially sources are chosen by looking at sources in circle with diameter of the FOV
# centred around the phase centre.
# Assuming the phase centre is at zenith
def main():
    # np.set_printoptions(linewidth=np.inf)
    random.seed(0)

    if len(sys.argv) < 2:
        sys.exit("Please provide name of the config yaml file")

    config = sys.argv[1]
    (
        ra_ph,
        dec_ph,
        T_sys,
        start_freq,
        end_freq,
        D,
        channel_width,
        srclist_dir,
        metafits_path,
        output,
        telescope,
        int_time,
        obs_name,
    ) = get_config(config)

    Path(output).mkdir(parents=True, exist_ok=True)

    print_with_time(
        f"INPUT SETTINGS: ra={ra_ph} dec={dec_ph} T_sys={T_sys} channel_width={channel_width} start_freq={start_freq} end_freq={end_freq} D={D}"
    )

    sigma = CRB.get_rms(T_sys, channel_width, telescope, int_time)
    print_with_time(f"CALCULATED NOISE: {sigma}")
    cut_off = 5 * (sigma / np.sqrt(8256))
    print_with_time(f"CALCULATED CUT OFF FOR SOURCES: {cut_off}")

    # Get source list
    print_with_time(f"READING IN SOURCE LIST FROM: {srclist_dir}")
    t1 = time.time()
    source_list, source_types = sources.get_source_list(
        srclist_dir, ra_ph, dec_ph, cut_off, output
    )
    t2 = time.time()

    print_with_time(f"READING IN TOOK: {t2 - t1}s")
    print_with_time(f"NUMBER OF SOURCES: {len(source_list)}")
    print(source_list)
    # print_with_time(f"TOP 10 SOURCES IN LIST ORDERED BY BRIGHTNESS")
    # sorted_source_list = source_list[source_list[:, 2].argsort()]
    # print(sorted_source_list[-10:])

    print_with_time(f"CREATING BASELINES")
    if telescope == "mwa":
        baseline_lengths = CRB.create_MWA_baselines(metafits_path)
        num_ant = 128
    elif telescope == "ska":
        # TODO:
        baseline_lengths = CRB.create_MWA_baselines(metafits_path)
        num_ant = 128

    # Create arrays that don't need to be created multiple times
    k_perp = power.create_k_perp(4000, 20)
    # NOTE: start_freq and end_freq here are the EDGES, could be redefined
    # to the centres of the starting and end channels
    # freq_array_edges = np.arange(start_freq, end_freq, channel_width)
    # freq_array = (freq_array_edges[:-1] + freq_array_edges[1:]) / 2.0

    num_freq = int(np.floor((end_freq - start_freq) / channel_width))
    # Ensure odd number of frequencies
    if np.mod(num_freq, 2) == 0:
        num_freq += 1

    freq_array = np.linspace(start_freq, end_freq, num_freq)

    pows = list()
    start_time = time.time()
    # Loop over frequencies and calculate the visibility uncertainties for each frequency
    for i in range(0, len(freq_array)):
        loop_start = time.time()
        freq = freq_array[i]
        lamb = const.c / freq
        print_with_time(f"==================== FREQUENCY: {freq} ====================")

        fov_sources, fov_source_types = sources.fov_cut(
            source_list, source_types, lamb, D
        )
        # print("UNSORTED")
        # print(f"{len(fov_sources)}, {len(fov_source_types)}")
        # for i in range(0, len(fov_sources)):
        #     print(
        #         f"{fov_sources[i, 0]} {fov_sources[i, 1]} {fov_sources[i, 2]} {fov_source_types[i]}"
        #     )

        sorted_fov = fov_sources[fov_sources[:, 2].argsort()]
        sorted_fov_source_types = fov_source_types[fov_sources[:, 2].argsort()]
        # print("SORTED")
        # print(f"{len(sorted_fov)}, {len(sorted_fov_source_types)}")
        # for i in range(0, len(fov_sources)):
        #     print(
        #         f"{sorted_fov[i, 0]} {sorted_fov[i, 1]} {sorted_fov[i, 2]} {sorted_fov_source_types[i]}"
        #     )
        print_with_time(f"NUMBER OF SOURCES: {len(fov_sources)}")

        print_with_time("CALCULATING BEAM")
        if telescope == "mwa":
            l_arr, m_arr, beam = CRB.beam_form_mwa(0, 90, lamb, D, output)
        elif telescope == "ska":
            l_arr, m_arr, beam = CRB.beam_form_ska(0, 90, lamb, D, output)

        print_with_time("ATTENUATING WITH BEAM")
        sorted_source_list = CRB.attenuate(sorted_fov, beam, l_arr, m_arr, output)
        # print(sorted_source_list)

        # Calculate FIM
        print_with_time("CALCULATING THE CRB")
        covar_mat = CRB.calculate_fim(
            sorted_source_list,
            baseline_lengths,
            num_ant,
            lamb,
            freq,
            sigma,
            output,
            sorted_fov_source_types,
        )

        mean_gain_unc = np.mean(np.sqrt(np.diag(covar_mat)))
        print_with_time(f"MEAN GAIN UNCERTAINTY: {mean_gain_unc}")

        # Propagate errors into visibilities
        print_with_time("PROPAGATING INTO VISIBILITIES")
        vis_uncertainties = propagate.propagate(
            baseline_lengths, sorted_source_list, covar_mat, lamb
        )

        mean_vis_unc = np.mean(vis_uncertainties)
        print_with_time(f"AVERAGE VISIBILITY ERROR {mean_vis_unc}")

        print_with_time("REALIZE ERRORS")
        realised_unc = propagate.realise(vis_uncertainties)

        print_with_time("BINNING ERRORS")
        vis_mat, u_arr, v_arr = power.uv_bin(
            lamb, realised_unc, baseline_lengths, output
        )

        # Calculate average power in annuli for this particular frequency
        print_with_time("ANGULAR POWER SPECTRUM")
        pow = power.power_bin(vis_mat, u_arr, v_arr, k_perp, output)
        pows.append(pow)

        loop_end = time.time()
        print_with_time(f"TOTAL RUN TIME FOR THIS FREQ: {loop_end - loop_start}")

    pows = np.array(pows)
    pows_fft = np.fft.fft(pows, axis=0)

    # freq_array_fft = np.abs(np.fft.fftfreq(freq_array))

    # Folding the FT, DC mode at first index.
    mid = int(np.ceil(num_freq / 2.0))
    # folded = freq_array_fft[0:mid]
    folded_pow = pows_fft[0:mid, :]
    folded_pow[1:, :] = (folded_pow[1:, :] + np.flip(pows_fft[mid:, :], axis=0)) / 2.0
    # folded[1:] = folded[1:] + np.flip(freq_array_fft[mid:])

    end_time = time.time()
    total = end_time - start_time
    print_with_time(f"TOTAL RUN TIME: {total}")

    save_hdf5(
        k_perp,
        folded_pow,
        output,
        telescope,
        channel_width,
        start_freq,
        end_freq,
        num_freq,
        ra_ph,
        dec_ph,
        obs_name,
    )
    # np.savetxt(output + "/power.txt", folded_pow, fmt="%1.4e")
    # plt.pcolormesh(k_perp, folded, np.abs(folded_pow), norm="log")
    # plt.yscale("log")
    # # plt.pcolormesh(np.abs(folded_pow))
    # plt.colorbar()
    # plt.show()
    # brightest = np.max(sorted_source_list)
    # Save pointing ra, pointing dec, mean CRB, num sources in FOV, brightest source in FOV
    # with open("output_" + telescope + ".txt", "w") as f:
    #     f.write(
    #         f"{'ra':>15s} {'dec':>15s} {'CRB':>15s} {'num src':>15s} {'brightest':>15s}\n"
    #     )
    #     f.write(
    #         f"{ra_ph:15f} {dec_ph:15f} {mean_CRB:15.5e} {len(source_list):15d} {brightest:15.5f}\n"
    #     )

    # Save visibility uncertainties
    # with open("vis_unc_" + telescope + ".txt", "w") as f:
    #     for row in vis_uncertainties:
    #         f.write(str(row) + "\n")


if __name__ == "__main__":
    main()
