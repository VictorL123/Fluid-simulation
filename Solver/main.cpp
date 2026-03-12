#include "solver.h"
#include <iostream>
#include <string>
#include <filesystem>
#include <cmath>
#include <fstream>

// ============================================================
//  Configuration
//  Change these values to experiment with different setups.
// ============================================================
static const int    N              = 41;      // Grid resolution (41x41 is a good start)
static const double NU             = 0.01;    // Viscosity  ->  Re = 1/nu = 100
static const double DT             = 0.001;   // Timestep   ->  must satisfy CFL condition
static const int    TOTAL_STEPS    = 15000;    // Total timesteps to simulate
static const int    OUTPUT_EVERY   = 100;     // Write CSV every N steps
static const int    PRESSURE_ITERS = 50;      // Jacobi iterations per step
static const std::string OUTPUT_DIR = "output";

// Probe point — records u,v at this cell every timestep for signal analysis
// Default: centre of the cavity
static const int PROBE_I = N / 2;
static const int PROBE_J = N / 2;

// ============================================================
//  Helpers
// ============================================================

// Check if simulation has gone unstable (any NaN or very large value)
bool is_unstable(const Grid& g) {
    for (double val : g.u) {
        if (std::isnan(val) || std::abs(val) > 1e6) return true;
    }
    return false;
}

// Print a progress update to stdout
void print_progress(int step, int total, const Grid& g) {
    // Find max velocity magnitude for monitoring
    double max_speed = 0.0;
    for (int i = 0; i < g.N*g.N; i++) {
        double speed = std::sqrt(g.u[i]*g.u[i] + g.v[i]*g.v[i]);
        if (speed > max_speed) max_speed = speed;
    }
    std::cout << "[step " << std::setw(5) << step << "/" << total << "]"
              << "  max_speed=" << std::fixed << std::setprecision(4) << max_speed
              << "\n";
}

// Zero-padded filename, e.g. step 500 -> "output/step_00500.csv"
std::string output_filename(int step) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%s/step_%05d.csv", OUTPUT_DIR.c_str(), step);
    return std::string(buf);
}

// ============================================================
//  main
// ============================================================
int main() {
    std::cout << "=== Lid-Driven Cavity Solver ===\n\n";

    // --- Create output directory ---
    std::filesystem::create_directories(OUTPUT_DIR);

    // --- Set up grid ---
    Grid g;
    g.N  = N;
    g.dt = DT;
    g.nu = NU;

    initialise(g);
    check_stability(g);
    std::cout << "\n";

    // --- Write initial state ---
    write_csv(g, output_filename(0));

    // --- Open probe file for time series recording ---
    std::ofstream probe_file(OUTPUT_DIR + "/probe_timeseries.csv");
    probe_file << "step,t,u,v,speed\n";

    // ============================================================
    //  Main time-stepping loop
    //
    //  Each call to step() does:
    //    1. apply_boundary()             — set lid + wall BCs
    //    2. compute_intermediate_velocity() — advect + diffuse
    //    3. solve_pressure()             — Poisson solve
    //    4. correct_velocity()           — project to div-free
    // ============================================================
    for (int s = 1; s <= TOTAL_STEPS; s++) {

        step(g, PRESSURE_ITERS);

        // --- Record probe point every timestep ---
        {
            double u_p = g.u[g.idx(PROBE_I, PROBE_J)];
            double v_p = g.v[g.idx(PROBE_I, PROBE_J)];
            double sp  = std::sqrt(u_p*u_p + v_p*v_p);
            double t   = s * g.dt;
            probe_file << s << "," << t << "," << u_p << "," << v_p << "," << sp << "\n";
        }

        // --- Stability check ---
        if (is_unstable(g)) {
            std::cerr << "\n[ERROR] Simulation went unstable at step " << s << "!\n";
            std::cerr << "        Try reducing dt (currently " << DT << ")\n";
            std::cerr << "        or reducing N (currently " << N << ")\n";
            return 1;
        }

        // --- Progress output ---
        if (s % OUTPUT_EVERY == 0) {
            print_progress(s, TOTAL_STEPS, g);
            write_csv(g, output_filename(s));
        }
    }

    std::cout << "\nSimulation complete. Output written to ./" << OUTPUT_DIR << "/\n";
    std::cout << "Run validate/plot.py to visualise results.\n";

    return 0;
}
