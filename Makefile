# Detect OS
UNAME_S := $(shell uname -s)

CFLAGS = -O3 -Wall
LDFLAGS = -lm
SRC = blocked_matmul.c
TARGET = blocked_matmul

# OpenMP-specific flags for macOS
ifeq ($(UNAME_S), Darwin)
	OMP_INCLUDE = -I/opt/homebrew/opt/libomp/include
	OMP_LIB = -L/opt/homebrew/opt/libomp/lib
	OMP_FLAGS = -Xpreprocessor -fopenmp -DUSE_OPENMP $(OMP_INCLUDE)
	OMP_LDFLAGS = $(OMP_LIB) -lomp
	CC = clang
else
	OMP_FLAGS = -fopenmp -DUSE_OPENMP
	OMP_LDFLAGS = -fopenmp
	CC = clang
endif

all: $(TARGET)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) -lm

omp:
	$(CC) $(CFLAGS) $(OMP_FLAGS) -o $(TARGET)_omp $(SRC) $(OMP_LDFLAGS) $(LDFLAGS)

clean:
	rm -f $(TARGET) $(TARGET)_omp
