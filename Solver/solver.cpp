#include "solver.h"
#include <cmath>
#include <stdexcept>
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif

// ============================================================
//  initialise
// ============================================================
void initialise(Grid& g) {
    // Checks grid must be at least 3x3
    if (g.N < 3) throw std::invalid_argument("N must be >= 3");

    // Calculate spacing between grid points
    g.dx = 1.0 / (g.N - 1);
    g.dy = 1.0 / (g.N - 1);

    // Initialise array
    int total = g.N * g.N;
    g.u.assign(total, 0.0);
    g.v.assign(total, 0.0);
    g.p.assign(total, 0.0);

    g.u_star.assign(total, 0.0);
    g.v_star.assign(total, 0.0);
    g.p_new.assign(total, 0.0);

#ifdef _OPENMP
    std::cout << "[init] Grid " << g.N << "x" << g.N
              << "  dx=" << g.dx
              << "  dt=" << g.dt
              << "  nu=" << g.nu
              << "  Re=" << (1.0 / g.nu)
              << "  threads=" << omp_get_max_threads() << "\n";
#else
    std::cout << "[init] Grid " << g.N << "x" << g.N
              << "  dx=" << g.dx
              << "  dt=" << g.dt
              << "  nu=" << g.nu
              << "  Re=" << (1.0 / g.nu)
              << "  threads=1 (serial build)\n";
#endif
}

// ============================================================
//  apply_boundary for Lid-driven Cavity setup
//  Called at the START of every timestep, serially.
//
// ============================================================
void apply_boundary(Grid& g) {
    int N = g.N;
    for (int k = 0; k < N; k++) {
        g.u[g.idx(N-1, k)] =  1.0; // Top lid moves right at speed 1
        g.v[g.idx(N-1, k)] =  0.0; // Top lid doesn't move up/down
        g.u[g.idx(0,   k)] =  0.0; // Bottom wall has no movement
        g.v[g.idx(0,   k)] =  0.0;
        g.u[g.idx(k,   0)] =  0.0; // Left wall has no movement
        g.v[g.idx(k,   0)] =  0.0;
        g.u[g.idx(k, N-1)] =  0.0; // Right wall has no movement
        g.v[g.idx(k, N-1)] =  0.0;
    }
}

// ============================================================
//  compute_intermediate_velocity
//
//  Must be called from INSIDE an existing omp parallel region.
//  Uses #pragma omp for (not parallel for) to participate in
//  the thread team created in step() — avoids spawning threads
//  on every call.
// ============================================================
void compute_intermediate_velocity(Grid& g) {
    int    N  = g.N;
    double dx = g.dx;
    double dy = g.dy;
    double dt = g.dt;
    double nu = g.nu;

    // Parallel copy into scratch arrays
    #pragma omp for schedule(static) nowait
    for (int k = 0; k < N * N; k++) {
        g.u_star[k] = g.u[k];
        g.v_star[k] = g.v[k];
    }
    #pragma omp barrier  // ensure copy done before compute

    // Advection + diffusion — each row independent, safe to parallelise
    #pragma omp for schedule(static)
    for (int i = 1; i < N-1; i++) {
        for (int j = 1; j < N-1; j++) {

            double u_c = g.u[g.idx(i, j)];
            double v_c = g.v[g.idx(i, j)];

            // Upwind advection
            double dudx = (u_c >= 0.0)
                ? (u_c                   - g.u[g.idx(i,   j-1)]) / dx
                : (g.u[g.idx(i,   j+1)] - u_c                  ) / dx;
            double dudy = (v_c >= 0.0)
                ? (u_c                   - g.u[g.idx(i-1, j  )]) / dy
                : (g.u[g.idx(i+1, j  )] - u_c                  ) / dy;
            double dvdx = (u_c >= 0.0)
                ? (v_c                   - g.v[g.idx(i,   j-1)]) / dx
                : (g.v[g.idx(i,   j+1)] - v_c                  ) / dx;
            double dvdy = (v_c >= 0.0)
                ? (v_c                   - g.v[g.idx(i-1, j  )]) / dy
                : (g.v[g.idx(i+1, j  )] - v_c                  ) / dy;

            // Central difference Laplacian (diffusion)
            double lap_u =
                (g.u[g.idx(i,   j+1)] - 2.0*u_c + g.u[g.idx(i,   j-1)]) / (dx*dx) +
                (g.u[g.idx(i+1, j  )] - 2.0*u_c + g.u[g.idx(i-1, j  )]) / (dy*dy);
            double lap_v =
                (g.v[g.idx(i,   j+1)] - 2.0*v_c + g.v[g.idx(i,   j-1)]) / (dx*dx) +
                (g.v[g.idx(i+1, j  )] - 2.0*v_c + g.v[g.idx(i-1, j  )]) / (dy*dy);

            // Combine effects to get intermediate velocity
            g.u_star[g.idx(i,j)] = u_c + dt * (-u_c*dudx - v_c*dudy + nu*lap_u);
            g.v_star[g.idx(i,j)] = v_c + dt * (-u_c*dvdx - v_c*dvdy + nu*lap_v);
        }
    }
    // Implicit barrier at end of omp for — u_star/v_star fully written
}

// ============================================================
//  solve_pressure
//
//  Jacobi iteration for the pressure Poisson equation.
//
//  KEY OPTIMISATION vs previous version:
//  The thread team is kept alive across ALL niter Jacobi
//  iterations. Only one thread handles the swap and boundary
//  update per iteration (#pragma omp single). This removes
//  the massive overhead of spawning threads 50x per timestep.
//
//  Must be called from inside an existing omp parallel region.
// ============================================================
void solve_pressure(Grid& g, int niter) {
    int    N  = g.N;
    double dx = g.dx;
    double dy = g.dy;
    double dt = g.dt;

    for (int iter = 0; iter < niter; iter++) {

        // All threads compute their chunk of the Jacobi update
        #pragma omp for schedule(static)
        for (int i = 1; i < N-1; i++) {
            for (int j = 1; j < N-1; j++) {
                // Calculates divergence at each point
                double div =
                    (g.u_star[g.idx(i, j+1)] - g.u_star[g.idx(i, j-1)]) / (2.0*dx) +
                    (g.v_star[g.idx(i+1, j)] - g.v_star[g.idx(i-1, j)]) / (2.0*dy);

                // New pressure is the average of its four neighbours adjusted by the divergence
                g.p_new[g.idx(i,j)] = 0.25 * (
                    g.p[g.idx(i,   j+1)] +
                    g.p[g.idx(i,   j-1)] +
                    g.p[g.idx(i+1, j  )] +
                    g.p[g.idx(i-1, j  )] -
                    (dx * dx) * div / dt
                );
            }
        }
        // Implicit barrier — all threads done before swap

        // One thread does boundary update and swap.
        // omp single has implicit barrier at end so all threads wait here before starting the next Jacobi iteration.
        #pragma omp single
        {
            // No pressure gradient at the wall
            for (int k = 0; k < N; k++) {
                g.p_new[g.idx(0,   k)] = g.p_new[g.idx(1,   k)];
                g.p_new[g.idx(N-1, k)] = g.p_new[g.idx(N-2, k)];
                g.p_new[g.idx(k,   0)] = g.p_new[g.idx(k,   1)];
                g.p_new[g.idx(k, N-1)] = g.p_new[g.idx(k, N-2)];
            }
            std::swap(g.p, g.p_new);
        }
    }
}

// ============================================================
//  correct_velocity
//
//  Project u*, v* onto divergence-free space.
//  Must be called inside an existing omp parallel region.
// ============================================================
void correct_velocity(Grid& g) {
    int    N  = g.N;
    double dx = g.dx;
    double dy = g.dy;
    double dt = g.dt;

    #pragma omp for schedule(static)
    for (int i = 1; i < N-1; i++) {
        for (int j = 1; j < N-1; j++) {
            double dpdx = (g.p[g.idx(i, j+1)] - g.p[g.idx(i, j-1)]) / (2.0 * dx);
            double dpdy = (g.p[g.idx(i+1, j)] - g.p[g.idx(i-1, j)]) / (2.0 * dy);
            g.u[g.idx(i,j)] = g.u_star[g.idx(i,j)] - dt * dpdx;
            g.v[g.idx(i,j)] = g.v_star[g.idx(i,j)] - dt * dpdy;
        }
    }
}

// ============================================================
//  step  —  one full timestep
//
//  The #pragma omp parallel region lives HERE, wrapping all
//  three parallel sub-steps. Threads are spawned ONCE per
//  timestep. Each sub-function uses #pragma omp for to
//  participate in this thread team without re-spawning.
//
//  apply_boundary is serial (boundary is tiny — N cells,
//  not N^2) so it runs before the parallel region opens.
// ============================================================
void step(Grid& g, int pressure_iters) {
    apply_boundary(g); // Step 1 Lock wall velocities

    // Threads are only spawned once per timestamp
    #pragma omp parallel
    {
        compute_intermediate_velocity(g); // Step 2 Advect + diffuse
        solve_pressure(g, pressure_iters); // Step 3 Solve pressure with 50 iterations
        correct_velocity(g); // Project to divergence-free
    }
}

// ============================================================
//  check_stability
// ============================================================
void check_stability(const Grid& g) {
    // Apply stability conditions for numerical methods
    double cfl_limit  = g.dx / 1.0;
    double diff_limit = (g.dx * g.dx) / (2.0 * g.nu);
    double safe_dt    = 0.5 * std::min(cfl_limit, diff_limit);

    std::cout << "[stability] CFL limit:       dt < " << cfl_limit  << "\n";
    std::cout << "[stability] Diffusion limit: dt < " << diff_limit << "\n";
    std::cout << "[stability] Recommended dt:       " << safe_dt    << "\n";
    std::cout << "[stability] Your dt:               " << g.dt       << "\n";

    if (g.dt > cfl_limit || g.dt > diff_limit) {
        std::cout << "[stability] WARNING: dt may be too large — simulation could blow up!\n";
    } else {
        std::cout << "[stability] dt looks safe.\n";
    }
}
