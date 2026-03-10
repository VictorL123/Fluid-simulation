#include "solver.h"
#include <cmath>
#include <stdexcept>
#include <iostream>

// ============================================================
//  initialise
// ============================================================
void initialise(Grid& g) {
    if (g.N < 3) throw std::invalid_argument("N must be >= 3");

    g.dx = 1.0 / (g.N - 1);
    g.dy = 1.0 / (g.N - 1);

    int total = g.N * g.N;
    g.u.assign(total, 0.0);
    g.v.assign(total, 0.0);
    g.p.assign(total, 0.0);

    g.u_star.assign(total, 0.0);
    g.v_star.assign(total, 0.0);
    g.p_new.assign(total, 0.0);

    std::cout << "[init] Grid " << g.N << "x" << g.N
              << "  dx=" << g.dx
              << "  dt=" << g.dt
              << "  nu=" << g.nu
              << "  Re=" << (1.0 / g.nu) << "\n";
}

// ============================================================
//  apply_boundary
//  Called at the START of every timestep.
//
//  Boundary conditions:
//    Top lid  (i = N-1):  u = 1.0,  v = 0   <- moves right
//    Bottom   (i = 0):    u = 0,    v = 0   <- no-slip wall
//    Left     (j = 0):    u = 0,    v = 0   <- no-slip wall
//    Right    (j = N-1):  u = 0,    v = 0   <- no-slip wall
// ============================================================
void apply_boundary(Grid& g) {
    int N = g.N;

    for (int k = 0; k < N; k++) {
        // Top lid — slides right at unit velocity
        g.u[g.idx(N-1, k)] =  1.0;
        g.v[g.idx(N-1, k)] =  0.0;

        // Bottom wall
        g.u[g.idx(0, k)]   =  0.0;
        g.v[g.idx(0, k)]   =  0.0;

        // Left wall
        g.u[g.idx(k, 0)]   =  0.0;
        g.v[g.idx(k, 0)]   =  0.0;

        // Right wall
        g.u[g.idx(k, N-1)] =  0.0;
        g.v[g.idx(k, N-1)] =  0.0;
    }
}

// ============================================================
//  compute_intermediate_velocity
//
//  Computes u*, v* by applying advection and diffusion,
//  but NOT the pressure gradient (that comes later).
//
//  Advection scheme: first-order upwind (stable, simple)
//  Diffusion scheme: central differences
//
//  Only updates INTERIOR cells (i=1..N-2, j=1..N-2).
//  Boundary cells are set by apply_boundary().
// ============================================================
void compute_intermediate_velocity(Grid& g) {
    int    N  = g.N;
    double dx = g.dx;
    double dy = g.dy;
    double dt = g.dt;
    double nu = g.nu;

    // Copy current velocity into u_star/v_star as starting point
    // so boundary cells are already correct in the scratch arrays
    g.u_star = g.u;
    g.v_star = g.v;

    for (int i = 1; i < N-1; i++) {
        for (int j = 1; j < N-1; j++) {

            double u_c = g.u[g.idx(i, j)];   // u at centre
            double v_c = g.v[g.idx(i, j)];   // v at centre

            // --- Advection: upwind scheme ---
            // du/dx: use left neighbour if u>0, right if u<0
            double dudx = (u_c >= 0.0)
                ? (u_c              - g.u[g.idx(i,   j-1)]) / dx
                : (g.u[g.idx(i,   j+1)] - u_c             ) / dx;

            // du/dy: use bottom neighbour if v>0, top if v<0
            double dudy = (v_c >= 0.0)
                ? (u_c              - g.u[g.idx(i-1, j  )]) / dy
                : (g.u[g.idx(i+1, j  )] - u_c             ) / dy;

            // dv/dx
            double dvdx = (u_c >= 0.0)
                ? (v_c              - g.v[g.idx(i,   j-1)]) / dx
                : (g.v[g.idx(i,   j+1)] - v_c             ) / dx;

            // dv/dy
            double dvdy = (v_c >= 0.0)
                ? (v_c              - g.v[g.idx(i-1, j  )]) / dy
                : (g.v[g.idx(i+1, j  )] - v_c             ) / dy;

            // --- Diffusion: Laplacian (central differences) ---
            double lap_u =
                (g.u[g.idx(i,   j+1)] - 2.0*u_c + g.u[g.idx(i,   j-1)]) / (dx*dx) +
                (g.u[g.idx(i+1, j  )] - 2.0*u_c + g.u[g.idx(i-1, j  )]) / (dy*dy);

            double lap_v =
                (g.v[g.idx(i,   j+1)] - 2.0*v_c + g.v[g.idx(i,   j-1)]) / (dx*dx) +
                (g.v[g.idx(i+1, j  )] - 2.0*v_c + g.v[g.idx(i-1, j  )]) / (dy*dy);

            // --- Update: explicit Euler timestep ---
            g.u_star[g.idx(i,j)] = u_c + dt * (-u_c*dudx - v_c*dudy + nu*lap_u);
            g.v_star[g.idx(i,j)] = v_c + dt * (-u_c*dvdx - v_c*dvdy + nu*lap_v);
        }
    }
}

// ============================================================
//  solve_pressure
//
//  Solves the pressure Poisson equation using Jacobi iteration:
//
//    ∇²P = (1/dt) * ∇·u*
//
//  This enforces the incompressibility constraint.
//  More iterations = more accurate, but slower.
//  50 iterations is a practical default for Re <= 400.
//
//  Neumann boundary condition on all walls: dP/dn = 0
//  (pressure gradient normal to wall = 0)
// ============================================================
void solve_pressure(Grid& g, int niter) {
    int    N  = g.N;
    double dx = g.dx;
    double dy = g.dy;
    double dt = g.dt;

    for (int iter = 0; iter < niter; iter++) {

        // --- Interior cells: Jacobi update ---
        for (int i = 1; i < N-1; i++) {
            for (int j = 1; j < N-1; j++) {

                // Divergence of intermediate velocity field
                double div =
                    (g.u_star[g.idx(i, j+1)] - g.u_star[g.idx(i, j-1)]) / (2.0*dx) +
                    (g.v_star[g.idx(i+1, j)] - g.v_star[g.idx(i-1, j)]) / (2.0*dy);

                // Jacobi update for pressure
                // Derived from 5-point Laplacian stencil
                g.p_new[g.idx(i,j)] = 0.25 * (
                    g.p[g.idx(i,   j+1)] +
                    g.p[g.idx(i,   j-1)] +
                    g.p[g.idx(i+1, j  )] +
                    g.p[g.idx(i-1, j  )] -
                    (dx * dx) * div / dt   // dx==dy so dx^2 = dy^2
                );
            }
        }

        // --- Boundary conditions: dP/dn = 0 (copy from interior neighbour) ---
        for (int k = 0; k < N; k++) {
            g.p_new[g.idx(0,   k)] = g.p_new[g.idx(1,     k)]; // bottom
            g.p_new[g.idx(N-1, k)] = g.p_new[g.idx(N-2,   k)]; // top
            g.p_new[g.idx(k,   0)] = g.p_new[g.idx(k,     1)]; // left
            g.p_new[g.idx(k, N-1)] = g.p_new[g.idx(k,   N-2)]; // right
        }

        // Swap p and p_new for next iteration
        std::swap(g.p, g.p_new);
    }
}

// ============================================================
//  correct_velocity
//
//  Project u*, v* onto divergence-free space using:
//    u = u* - dt * dP/dx
//    v = v* - dt * dP/dy
// ============================================================
void correct_velocity(Grid& g) {
    int    N  = g.N;
    double dx = g.dx;
    double dy = g.dy;
    double dt = g.dt;

    for (int i = 1; i < N-1; i++) {
        for (int j = 1; j < N-1; j++) {
            // Central difference pressure gradient
            double dpdx = (g.p[g.idx(i, j+1)] - g.p[g.idx(i, j-1)]) / (2.0 * dx);
            double dpdy = (g.p[g.idx(i+1, j)] - g.p[g.idx(i-1, j)]) / (2.0 * dy);

            g.u[g.idx(i,j)] = g.u_star[g.idx(i,j)] - dt * dpdx;
            g.v[g.idx(i,j)] = g.v_star[g.idx(i,j)] - dt * dpdy;
        }
    }
}

// ============================================================
//  step  —  one full timestep
// ============================================================
void step(Grid& g, int pressure_iters) {
    apply_boundary(g);
    compute_intermediate_velocity(g);
    solve_pressure(g, pressure_iters);
    correct_velocity(g);
}

// ============================================================
//  check_stability
//  Prints CFL and diffusion stability limits.
//  Call this before your main loop.
// ============================================================
void check_stability(const Grid& g) {
    double cfl_limit  = g.dx / 1.0;              // u_max assumed = 1 (lid velocity)
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
