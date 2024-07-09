import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from numba import jit, prange


@jit(nopython=True, cache=True)
def find_uv_index(u, v, u_arr, v_arr):
    """Function for determining the correct indices

    Parameters
    ----------
    - u: `float`
        u coordinate
    - v: `float`
        v coordinate
    - u_arr: `np.array`
        array of u values in grid
    - v_arr: `np.array`
        array of v values in grid

    Returns
    -------
    - u_ind: `int`
        u index
    - v_ind: `int`
        v_index
    """
    return np.argmin(np.abs(u - u_arr)), np.argmin(np.abs(v - v_arr))


@jit(nopython=True, cache=True)
def uv_bin(lamb, vis, baseline_lengths, output):
    """Function for binning visibility points

    Parameters
    ----------
    - lamb: `float`
        Observing wavelength
    - vis: `np.array`
        Matrix of visibilities, or uncertainties to make power spectra
    - baseline_lengths: `np.array`
        Matrix of baselines
    """

    num_ant = baseline_lengths.shape[0]

    # Create uv-grid
    baselines = baseline_lengths / lamb

    max_u = np.ceil(np.max(baselines[:, :, 0]))
    max_v = np.ceil(np.max(baselines[:, :, 1]))
    max = np.max(np.array([int(max_u), int(max_v)]))
    max_dist = np.sqrt(2 * max**2)
    du = 10

    u_num_cells = int(np.ceil(max_dist * 2 / du))
    if np.mod(u_num_cells, 2) == 0:
        u_num_cells += 1

    u_arr = np.linspace(-max_dist, max_dist, int(u_num_cells))
    v_arr = np.linspace(-max_dist, max_dist, int(u_num_cells))
    vis_mat = np.zeros((u_num_cells, u_num_cells), dtype=np.complex64)
    weights = np.zeros((u_num_cells, u_num_cells))

    # NOTE: u is in the rows, so does that mean it's the y-axis?
    # Probably. Just a simple 90 degree rot anyway.
    for i in range(0, num_ant):
        for j in range(0, num_ant):
            u = baselines[i, j, 0]
            v = baselines[i, j, 1]

            u_ind, v_ind = find_uv_index(u, v, u_arr, v_arr)
            vis_mat[u_ind, v_ind] += vis[i, j]
            weights[u_ind, v_ind] += 1

    for i in range(vis_mat.shape[0]):
        for j in range(vis_mat.shape[1]):
            if weights[i, j] != 0:
                vis_mat[i, j] /= weights[i, j]

    # vis_mat[weights != 0] /= weights[weights != 0]

    # plt.imshow((np.abs(vis_mat)), origin="lower")
    # plt.xlabel("u")
    # plt.ylabel("v")
    # plt.title("Visibility errors")
    # plt.savefig(output + "/vis_errors_no_arc.png", bbox_inches="tight")
    # plt.clf()

    return vis_mat, u_arr, v_arr


def create_k_perp(max_k, dk):
    """Function for creating k_perp

    Parameters
    ----------
    - max_k: `int`
        maximum k_perp value
    - dk: `float`
        annuli width

    Returns
    -------
    - k_perp: `np.array`
        array of k_perp
    """

    # Create k_perp grid
    cosmo = 1
    # max_k = np.max(u_arr) * cosmo
    max_k *= cosmo
    k_perp = np.arange(0, max_k, dk)

    return k_perp


@jit(nopython=True, cache=True)
def power_bin(vis_mat, u_arr, v_arr, k_perp, output):
    """Function to bin in the FT space

    Parameters
    ----------
    - vis_mat: `np.array`
        array of visibilities or whatever
    - u_arr: `np.array`
        array of u coords
    - v_arr: `np.array`
        array of v coords
    - k_perp: `np.array`
        array of k_perp
    - output: `string`
        path to directory to save files

    Returns
    -------
    None
    """

    pow = np.zeros(len(k_perp), dtype=np.complex64)

    for i in range(1, len(k_perp)):
        num_in_annuli = 0
        for j in range(0, len(u_arr)):
            u = u_arr[j]
            for k in range(0, len(v_arr)):
                v = v_arr[k]
                dist = np.sqrt(u**2 + v**2)
                if dist >= k_perp[i - 1] and dist < k_perp[i]:
                    num_in_annuli += 1
                    pow[i] += vis_mat[j, k]

        pow[i] /= num_in_annuli

    pow *= np.conjugate(pow)

    return pow

    # plt.plot(k_perp, pow)
    # plt.xlabel("k_perp")
    # plt.ylabel("power")
    # plt.savefig(output + "/angular_pow.png", bbox_inches="tight")
    # plt.clf()

    # Create visibility scatter plot again, but with the arcs.
    # for i in range(1, len(k_perp)):
    #     plt.gca().add_patch(
    #         patches.Circle(
    #             (0, 0), k_perp[i], facecolor=None, edgecolor="black", fill=False
    #         )
    #     )

    # plt.imshow(np.abs(vis_mat), origin="lower", extent=(-max_k, max_k, -max_k, max_k))
    # plt.xlim((-max_k, max_k))
    # plt.ylim((-max_k, max_k))
    # plt.xlabel("u")
    # plt.ylabel("v")
    # plt.title("Visibility errors")
    # plt.colorbar()
    # plt.savefig(output + "/vis_errors.png", bbox_inches="tight")
    # plt.clf()
