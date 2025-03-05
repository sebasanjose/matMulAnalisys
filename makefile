# Makefile for compiling matrix multiplication CUDA files

NVCC    := nvcc
CFLAGS  := -O3

# List your .cu source files
SOURCES := matMulNaive.cu matMulTiling.cu matMulCoarsening.cu

# Create a list of targets by replacing .cu with an empty string
TARGETS := $(SOURCES:.cu=)

all: $(TARGETS)

# Pattern rule: compile each .cu file into an executable with the same basename.
%: %.cu
	$(NVCC) $(CFLAGS) $< -o $@

clean:
	rm -f $(TARGETS)
