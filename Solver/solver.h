#pragma once
#include <vector>
#include <string>

// ============================================================
//  Grid
//  Holds all simulation state as flat 1D vectors.
//  Access cell (row i, col j) with index: i*N + j
//  i = 0 is bottom wall, i = N-1 is top lid
//  j = 0 is left wall,   j = N-1 is right wall
// ============================================================
struct Grid {
    int    N;       // number of grid points along each axis
    double dx;      // grid spacing in x  (= 1.0 / (N-1))
    double dy;      // grid spacing in y  (= 1.0 / (N-1))
    double dt;      // timestep
    double nu;      // kinematic viscosity  (Re = U*L/nu, U=L=1 => Re = 1/nu)

    std::vector<double> u;      // x-velocity at each cell
    std::vector<double> v;      // y-velocity at each cell
    std::vector<double> p;      // pressure at each cell

    // Scratch arrays — allocated once, reused every step
    std::vector<double> u_star; // intermediate x-velocity (before pressure correction)
    std::vector<double> v_star; // intermediate y-velocity
    std::vector<double> p_new;  // pressure after one Jacobi sweep

    // Inline helper: 2D index -> 1D index
    inline int idx(int i, int j) const { return i * N + j; }
};

// ============================================================
//  Core solver functions  (implemented in solver.cpp)
// ============================================================

// Allocate arrays and set everything to zero
void initialise(Grid& g);

// Apply no-slip walls and moving lid boundary conditions
void apply_boundary(Grid& g);

// Compute intermediate velocity u*, v* (advection + diffusion, no pressure)
void compute_intermediate_velocity(Grid& g);

// Iteratively solve pressure Poisson equation (Jacobi iterations)
// niter: number of Jacobi sweeps per timestep (50 is a good default)
void solve_pressure(Grid& g, int niter = 50);

// Correct u*, v* using pressure gradient -> divergence-free u, v
void correct_velocity(Grid& g);

// Advance simulation by one full timestep
// Calls: apply_boundary -> intermediate_velocity -> solve_pressure -> correct_velocity
void step(Grid& g, int pressure_iters = 50);

// ============================================================
//  I/O utilities  (implemented in io.cpp)
// ============================================================

// Write u, v, p fields to a CSV file for visualisation/validation
// Format: i, j, x, y, u, v, p   (one row per cell)
void write_csv(const Grid& g, const std::string& filepath);

// Print a simple stability check to stdout
// Warns if CFL condition is likely violated
void check_stability(const Grid& g);
