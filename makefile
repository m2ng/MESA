BUILD_DIR := ./build
SRC_DIR := ./src
EIGEN_DIR := $(HOME)/.local/include/eigen-3.3.7

all: make_build_dir
	g++ -Wall -std=c++17 -O2 -I$(EIGEN_DIR) -march=native -ffast-math -fopenmp -mavx -o $(BUILD_DIR)/mesa $(SRC_DIR)/mesa.cc
	
make_build_dir:
	mkdir -p $(BUILD_DIR)

debug: make_build_dir
	g++ -Wall -std=c++17 -O2 -I$(EIGEN_DIR) -march=native -ffast-math -fopenmp -DDEBUG_PRINT -mavx -o $(BUILD_DIR)/mesa_debug $(SRC_DIR)/mesa.cc

clean:
	rm -r $(BUILD_DIR)