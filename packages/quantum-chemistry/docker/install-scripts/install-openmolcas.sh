#!/bin/bash
# OpenMolcas CASPT2 Installation Script

set -e

echo "Installing OpenMolcas package..."

# Check if we're in Docker or native environment
if [ -f /.dockerenv ]; then
    echo "Running in Docker environment"
    INSTALL_PREFIX="/opt/molcas"
else
    echo "Running in native environment"
    INSTALL_PREFIX="$HOME/quantum-software/molcas"
    mkdir -p "$INSTALL_PREFIX"
fi

# Clone repository if not exists
if [ ! -d "$INSTALL_PREFIX" ]; then
    echo "Cloning OpenMolcas repository..."
    git clone https://gitlab.com/Molcas/OpenMolcas.git "$INSTALL_PREFIX"
fi

cd "$INSTALL_PREFIX"

# Initialize LAPACK submodule
echo "Initializing LAPACK submodule..."
git submodule update --init External/lapack

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo "Configuring OpenMolcas build..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX/install" \
    -DCMAKE_Fortran_FLAGS="-O3" \
    -DCMAKE_C_FLAGS="-O3" \
    -DCMAKE_CXX_FLAGS="-O3"

# Build and install
echo "Building OpenMolcas..."
make -j$(nproc) install

# Set up environment
echo "Setting up OpenMolcas environment..."
cat >> ~/.bashrc << EOF
export MOLCAS=$INSTALL_PREFIX/install
export MOLCAS_MEM=2000
export PATH=\$MOLCAS/bin:\$PATH
export LD_LIBRARY_PATH=\$MOLCAS/lib:\$LD_LIBRARY_PATH
EOF

# Test installation
echo "Testing OpenMolcas installation..."
export MOLCAS="$INSTALL_PREFIX/install"
export PATH="$MOLCAS/bin:$PATH"
if command -v pymolcas >/dev/null 2>&1; then
    echo "OpenMolcas installation successful!"
else
    echo "Warning: pymolcas executable not found, but installation may still be functional"
fi

echo "OpenMolcas installation completed!"
echo "Please run 'source ~/.bashrc' to load the environment variables."