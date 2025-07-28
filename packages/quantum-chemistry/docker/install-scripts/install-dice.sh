#!/bin/bash
# Dice SHCI Installation Script

set -e

echo "Installing Dice SHCI package..."

# Check if we're in Docker or native environment
if [ -f /.dockerenv ]; then
    echo "Running in Docker environment"
    INSTALL_PREFIX="/opt/dice"
else
    echo "Running in native environment"
    INSTALL_PREFIX="$HOME/quantum-software/dice"
    mkdir -p "$INSTALL_PREFIX"
fi

# Check for required packages
echo "Checking dependencies..."
if ! command -v mpicxx >/dev/null 2>&1; then
    echo "Error: MPI C++ compiler (mpicxx) not found. Please install OpenMPI or MPICH."
    exit 1
fi

# Clone repository if not exists
if [ ! -d "$INSTALL_PREFIX" ]; then
    echo "Cloning Dice repository..."
    git clone https://github.com/sanshar/Dice.git "$INSTALL_PREFIX"
fi

cd "$INSTALL_PREFIX"

# Create custom Makefile for the installation
echo "Creating Makefile for Dice build..."
cat > Makefile << 'EOF'
CXX = mpicxx
CXXFLAGS = -std=c++11 -O3 -DNDEBUG
BOOST_INCLUDE = /usr/include/boost
EIGEN_INCLUDE = /usr/include/eigen3
MPI_INCLUDE = /usr/include/mpi

# Detect system paths
BOOST_INCLUDE := $(shell find /usr/include /usr/local/include -name "boost" -type d 2>/dev/null | head -1)
EIGEN_INCLUDE := $(shell find /usr/include /usr/local/include -name "eigen3" -type d 2>/dev/null | head -1)

INCLUDES = -I$(BOOST_INCLUDE) -I$(EIGEN_INCLUDE) -I$(MPI_INCLUDE) -I.

SOURCES = $(wildcard *.cpp) $(wildcard */*.cpp)
OBJECTS = $(SOURCES:.cpp=.o)
TARGET = Dice

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ -lboost_serialization -lboost_mpi

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)

.PHONY: all clean
EOF

# Build Dice
echo "Building Dice..."
make -j$(nproc)

# Set up environment
echo "Setting up Dice environment..."
echo "export DICE_ROOT=$INSTALL_PREFIX" >> ~/.bashrc
echo "export PATH=$INSTALL_PREFIX:\$PATH" >> ~/.bashrc

# Test installation
echo "Testing Dice installation..."
if [ -f "$INSTALL_PREFIX/Dice" ]; then
    echo "Dice installation successful!"
    ls -la "$INSTALL_PREFIX/Dice"
else
    echo "Error: Dice executable not found"
    exit 1
fi

echo "Dice installation completed!"
echo "Please run 'source ~/.bashrc' to load the environment variables."