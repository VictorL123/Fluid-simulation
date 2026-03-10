#include "solver.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <cmath>

// ============================================================
//  write_csv
//
//  Writes the full field state to a CSV file.
//  Columns: i, j, x, y, u, v, p, speed
//
//  'speed' = sqrt(u^2 + v^2) — useful for visualisation
//
//  Usage example:
//    write_csv(g, "output/step_0500.csv");
// ============================================================
void write_csv(const Grid& g, const std::string& filepath) {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filepath);
    }

    // Header row
    file << "i,j,x,y,u,v,p,speed\n";
    file << std::fixed << std::setprecision(8);

    for (int i = 0; i < g.N; i++) {
        for (int j = 0; j < g.N; j++) {
            double x     = j * g.dx;
            double y     = i * g.dy;
            double u_val = g.u[g.idx(i, j)];
            double v_val = g.v[g.idx(i, j)];
            double p_val = g.p[g.idx(i, j)];
            double speed = std::sqrt(u_val*u_val + v_val*v_val);

            file << i     << ","
                 << j     << ","
                 << x     << ","
                 << y     << ","
                 << u_val << ","
                 << v_val << ","
                 << p_val << ","
                 << speed << "\n";
        }
    }

    file.close();
    std::cout << "[io] Written: " << filepath << "\n";
}
