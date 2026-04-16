#include "solver.h"
#include <iostream>
#include <string>
#include <filesystem>
#include <cmath>
#include <fstream>
#include <iomanip>

// ============================================================
//  Configuration — defaults, can be overridden via command line
//  Usage: ./fluid_sim_omp.exe [nu] [total_steps] [output_every]
//  Example: ./fluid_sim_omp.exe 0.01 15000 100
// ============================================================
static const int    N              = 41;
static const double NU             = 0.01;
static const double DT             = 0.001;
static const int    TOTAL_STEPS    = 15000;
static const int    OUTPUT_EVERY   = 100;
static const int    PRESSURE_ITERS = 50;
static const std::string OUTPUT_DIR = "output";

static const int PROBE_I = N / 2;
static const int PROBE_J = N / 2;

// ============================================================
//  Helpers
// ============================================================

// Checks if simulation has blown up
bool is_unstable(const Grid& g) {
    for (double val : g.u) {
        if (std::isnan(val) || std::abs(val) > 1e6) return true;
    }
    return false;
}

void print_progress(int step, int total, const Grid& g) {
    double max_speed = 0.0;
    for (int i = 0; i < g.N*g.N; i++) {
        double speed = std::sqrt(g.u[i]*g.u[i] + g.v[i]*g.v[i]);
        if (speed > max_speed) max_speed = speed;
    }
    std::cout << "[step " << std::setw(5) << step << "/" << total << "]"
              << "  max_speed=" << std::fixed << std::setprecision(4) << max_speed
              << "\n";
}

std::string output_filename(int step) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%s/step_%05d.csv", OUTPUT_DIR.c_str(), step);
    return std::string(buf);
}

// ============================================================
//  main
//  Accepts optional command line arguments:
//    argv[1] = nu          (viscosity, default 0.01)
//    argv[2] = total_steps (default 15000)
//    argv[3] = output_every (default 100)
//
//  This allows generate_data.py to call the solver with
//  different nu values WITHOUT recompiling each time.
// ============================================================
int main(int argc, char* argv[]) {
    std::cout << "=== Lid-Driven Cavity Solver ===\n\n";

    std::filesystem::create_directories(OUTPUT_DIR);

    // --- Set up grid with defaults ---
    Grid g;
    g.N  = N;
    g.dt = DT;
    g.nu = NU;
    int total_steps  = TOTAL_STEPS;
    int output_every = OUTPUT_EVERY;

    // --- Override from command line if provided ---
    if (argc > 1) g.nu        = std::stod(argv[1]);
    if (argc > 2) total_steps = std::stoi(argv[2]);
    if (argc > 3) output_every= std::stoi(argv[3]);

    std::cout << "[config] nu=" << g.nu
              << "  Re=" << (1.0/g.nu)
              << "  steps=" << total_steps << "\n";

    initialise(g);
    check_stability(g);
    std::cout << "\n";

    // --- Write initial state ---
    write_csv(g, output_filename(0));

    // --- Open probe file ---
    std::ofstream probe_file(OUTPUT_DIR + "/probe_timeseries.csv");
    probe_file << "step,t,u,v,speed\n";

    // ============================================================
    //  Main time-stepping loop
    // ============================================================
    for (int s = 1; s <= total_steps; s++) {

        step(g, PRESSURE_ITERS);

        // Record probe point
        {
            double u_p = g.u[g.idx(PROBE_I, PROBE_J)];
            double v_p = g.v[g.idx(PROBE_I, PROBE_J)];
            double sp  = std::sqrt(u_p*u_p + v_p*v_p);
            double t   = s * g.dt;
            probe_file << s << "," << t << "," << u_p << "," << v_p << "," << sp << "\n";
        }

        // Stability check
        if (is_unstable(g)) {
            std::cerr << "\n[ERROR] Simulation went unstable at step " << s << "!\n";
            std::cerr << "        Try reducing dt (currently " << g.dt << ")\n";
            return 1;
        }

        // Progress + CSV output
        if (s % output_every == 0) {
            print_progress(s, total_steps, g);
            write_csv(g, output_filename(s));
        }
    }

    std::cout << "\nSimulation complete. Output written to ./" << OUTPUT_DIR << "/\n";
    return 0;
}
