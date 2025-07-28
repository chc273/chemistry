#!/bin/bash
# Block2 DMRG Installation Script

set -e

echo "Installing Block2 DMRG package..."

# Check if we're in Docker or native environment
if [ -f /.dockerenv ]; then
    echo "Running in Docker environment"
    INSTALL_PREFIX="/opt/block2"
else
    echo "Running in native environment"
    INSTALL_PREFIX="$HOME/quantum-software/block2"
    mkdir -p "$INSTALL_PREFIX"
fi

# Check dependencies
echo "Checking dependencies..."
python3 -c "import pybind11; print('pybind11 version:', pybind11.__version__)" || {
    echo "Installing pybind11..."
    pip3 install "pybind11>=2.12"
}

# Clone repository if not exists
if [ ! -d "$INSTALL_PREFIX" ]; then
    echo "Cloning Block2 repository..."
    git clone https://github.com/block-hczhai/block2-preview.git "$INSTALL_PREFIX"
fi

cd "$INSTALL_PREFIX"

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo "Configuring Block2 build..."
cmake .. \
    -DUSE_MKL=ON \
    -DBUILD_LIB=ON \
    -DLARGE_BOND=ON \
    -DMPI=ON \
    -DUSE_COMPLEX=ON \
    -DUSE_SG=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-O3"

# Build
echo "Building Block2..."
make -j$(nproc)

# Set up environment
echo "Setting up Block2 environment..."
echo "export PYTHONPATH=$INSTALL_PREFIX/build:$INSTALL_PREFIX:\$PYTHONPATH" >> ~/.bashrc
echo "export PATH=$INSTALL_PREFIX/build:\$PATH" >> ~/.bashrc

# Test installation
echo "Testing Block2 installation..."
export PYTHONPATH="$INSTALL_PREFIX/build:$INSTALL_PREFIX:$PYTHONPATH"
python3 -c "import block2; print('Block2 installation successful!')" || {
    echo "Block2 installation test failed"
    exit 1
}

echo "Block2 installation completed successfully!"
echo "Please run 'source ~/.bashrc' to load the environment variables."