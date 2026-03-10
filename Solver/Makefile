# ============================================================
#  Makefile for Lid-Driven Cavity Solver
#  
#  Usage:
#    make          -> build serial version
#    make run      -> build and run
#    make clean    -> remove build artifacts
#
#  Phase 2 (OpenMP) will add:
#    make omp      -> build with OpenMP parallelisation
# ============================================================

CXX      = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra

# Source files
SRCS = main.cpp solver.cpp io.cpp
TARGET = fluid_sim

# ============================================================
#  Default: serial build
# ============================================================
all: $(TARGET)

$(TARGET): $(SRCS) solver.h
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRCS)
	@echo "Build successful: ./$(TARGET)"

# ============================================================
#  Run the simulation
# ============================================================
run: all
	./$(TARGET)

# ============================================================
#  OpenMP build  (Phase 2 — uncomment when ready)
#  Adds -fopenmp flag to enable #pragma omp parallel for
# ============================================================
# omp: $(SRCS) solver.h
# 	$(CXX) $(CXXFLAGS) -fopenmp -o $(TARGET)_omp $(SRCS)
# 	@echo "OpenMP build successful: ./$(TARGET)_omp"

# ============================================================
#  Clean
# ============================================================
clean:
	rm -f $(TARGET) $(TARGET)_omp
	rm -rf output/

.PHONY: all run clean
