#pragma OPENCL EXTENSION cl_khr_fp64 : enable

kernel void calculate_crb_kernel(const int n_ant, global const double
*baselines_x, global const double *baselines_y, const int n_sources, global
const double *source_intensities, global const double* source_l, global const
double* source_m, double lambda, double sigma, global double* result) {

    // `result` will hold the final FIM. Inversion will happen on the host.
    // Each thread will work on an individual FIM matrix element, no reduction needed

    /* size_t l_id = get_local_id(0); // Local ID of thread in work group */
    /* size_t l_size = get_local_size(0); // Size of work group */

    size_t g_id = get_global_id(0);

    double u_ab = baselines_x[g_id] / lambda;
    double v_ab = baselines_y[g_id] / lambda ;

    double2 s_ab = (double2)(0.0, 0.0);
    for (int i_source = 0; i_source < n_sources; i_source++) {
        double phase_arg = -2.0 * M_PI * (u_ab * source_l[i_source] + v_ab * source_m[i_source]);
        s_ab.x += source_intensities[i_source] * cos(phase_arg);
        s_ab.y += source_intensities[i_source] * sin(phase_arg);
    }

    result[g_id] = s_ab.x * s_ab.x + s_ab.y * s_ab.y;

    int a = (int)(g_id / n_ant);
    int b = (int)(g_id % n_ant);
    if (a == b) {
        result[g_id] *= 131;
    }

    result[g_id] *= 2.0/(sigma * sigma);
}
