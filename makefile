# Compiler and linker flags
CXX = g++
MPICXX = mpicxx

CXXFLAGS = -fopenmp -m64 -O2 -std=c++11 -I/home/xianglong/code/external_linux_reorganize/IntelMKL/2021.3/include
LDFLAGS = -L/home/xianglong/code/external_linux_reorganize/IntelMKL/2021.3/lib -Wl,--start-group -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -Wl,--end-group -lpthread -lm -ldl -liomp5 -lhwloc

# Targets
all: matmul

matmul: matmul.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

mpi_test: matmul_mpi

matmul_mpi: matmul_mpi.cpp
	$(MPICXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

bench_test: matmul_bench.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f matmul matmul_mpi bench_test

