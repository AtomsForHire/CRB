#ifdef SINGLE
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

typedef REAL  real_t;
typedef REAL2 real2_t;

kernel void calculate_crb_kernel(const int n_ant, global const real_t
*baselines_x, global const real_t *baselines_y, const int n_sources, global
const real_t *source_intensities, global const real_t* source_l, global const
real_t* source_m, real_t lambda, real_t sigma, global real_t* result) {

    // `result` will hold the final FIM. Inversion will happen on the host.
    // Each thread will work on an individual FIM matrix element, no reduction needed

    /* size_t l_id = get_local_id(0); // Local ID of thread in work group */
    /* size_t l_size = get_local_size(0); // Size of work group */

    size_t g_id = get_global_id(0);
    int i = (int)(g_id / (2 * n_ant));
    int j = (int)(g_id % (2 * n_ant));

    // Determine if thread is within the top-right or bottom-left region
    bool row_is_in_gains = (i < n_ant);
    bool col_is_in_gains = (j < n_ant);

    if ((row_is_in_gains && !col_is_in_gains) || (!row_is_in_gains && col_is_in_gains)) {
        result[g_id] = (real_t)0.0;
        return;
    }

    int ant_a = row_is_in_gains ? i : (i - n_ant);
    int ant_b = col_is_in_gains ? j : (j - n_ant);
    int baseline_idx = ant_a * n_ant + ant_b;

    real_t u_ab = baselines_x[baseline_idx] / lambda;
    real_t v_ab = baselines_y[baseline_idx] / lambda ;

    real_t diag_term_gains = (real_t)2.0 * ((real_t)n_ant + (real_t)1.0); 
    real_t diag_term_phases = (real_t)2.0 * ((real_t)n_ant - (real_t)1.0); 

    real2_t s_ab = (real2_t)(0.0, 0.0);
    for (int i_source = 0; i_source < n_sources; i_source++) {
        real_t phase_arg = -(real_t)2.0 * (real_t)M_PI * (u_ab * source_l[i_source] + v_ab * source_m[i_source]);
        //printf("source_l: %f, source_m: %f, phase arg: %f\n", source_l[i_source], source_m[i_source], phase_arg);
        s_ab.x += source_intensities[i_source] * cos(phase_arg);
        s_ab.y += source_intensities[i_source] * sin(phase_arg);
    }

    result[g_id] = s_ab.x * s_ab.x + s_ab.y * s_ab.y;

    bool is_diagonal = (i == j);

    if (row_is_in_gains && col_is_in_gains) {
        real_t factor = is_diagonal ? diag_term_gains : (real_t)2.0;
        result[g_id] *= factor;
    } else {
        real_t factor = is_diagonal ? diag_term_phases : (real_t)(-2.0);
        result[g_id] *= factor;
    }

    result[g_id] *= 2.0/(sigma * sigma);
}
