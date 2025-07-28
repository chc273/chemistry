#!/bin/bash
# Quantum Package CIPSI Installation Script

set -e

echo "Installing Quantum Package..."

# Check if we're in Docker or native environment
if [ -f /.dockerenv ]; then
    echo "Running in Docker environment"
    INSTALL_PREFIX="/opt/quantum-package"
else
    echo "Running in native environment"
    INSTALL_PREFIX="$HOME/quantum-software/quantum-package"
    mkdir -p "$INSTALL_PREFIX"
fi

# Check for required packages
echo "Checking dependencies..."
if ! command -v ocaml >/dev/null 2>&1; then
    echo "Error: OCaml not found. Please install OCaml and OPAM."
    exit 1
fi

# Install Ninja if not present
if ! command -v ninja >/dev/null 2>&1; then
    echo "Installing Ninja build system..."
    if [ -f /.dockerenv ]; then
        # In Docker, we can install system-wide
        wget -O /tmp/ninja.zip https://github.com/ninja-build/ninja/releases/latest/download/ninja-linux.zip
        unzip /tmp/ninja.zip -d /usr/local/bin
        rm /tmp/ninja.zip
    else
        # In native environment, install to user local
        mkdir -p "$HOME/.local/bin"
        wget -O /tmp/ninja.zip https://github.com/ninja-build/ninja/releases/latest/download/ninja-linux.zip
        unzip /tmp/ninja.zip -d "$HOME/.local/bin"
        rm /tmp/ninja.zip
        export PATH="$HOME/.local/bin:$PATH"
        echo "export PATH=\"$HOME/.local/bin:\$PATH\"" >> ~/.bashrc
    fi
fi

# Install IRPF90 if not present
if ! command -v irpf90 >/dev/null 2>&1; then
    echo "Installing IRPF90..."
    IRPF90_DIR="/tmp/irpf90-1.7.7"
    wget -O /tmp/irpf90.tar.gz https://github.com/scemama/irpf90/archive/refs/tags/v1.7.7.tar.gz
    tar -xzf /tmp/irpf90.tar.gz -C /tmp
    cd "$IRPF90_DIR"
    make
    if [ -f /.dockerenv ]; then
        cp bin/irpf90 /usr/local/bin/
    else
        mkdir -p "$HOME/.local/bin"
        cp bin/irpf90 "$HOME/.local/bin/"
        export PATH="$HOME/.local/bin:$PATH"
    fi
    rm -rf /tmp/irpf90*
fi

# Clone repository if not exists
if [ ! -d "$INSTALL_PREFIX" ]; then
    echo "Cloning Quantum Package repository..."
    git clone https://github.com/QuantumPackage/qp2.git "$INSTALL_PREFIX"
fi

cd "$INSTALL_PREFIX"

# Set up OCaml environment if not in Docker
if [ ! -f /.dockerenv ]; then
    if [ ! -d "$HOME/.opam" ]; then
        echo "Initializing OPAM..."
        opam init --disable-sandboxing -y
    fi
    eval $(opam env)
    opam install -y dune
fi

# Configure Quantum Package
echo "Configuring Quantum Package..."
./configure --install

# Build Quantum Package
echo "Building Quantum Package..."
source quantum_package.rc
ninja

# Set up environment
echo "Setting up Quantum Package environment..."
cat >> ~/.bashrc << EOF
export QP_ROOT=$INSTALL_PREFIX
if [ -f "\$QP_ROOT/quantum_package.rc" ]; then
    source \$QP_ROOT/quantum_package.rc
fi
EOF

# Test installation
echo "Testing Quantum Package installation..."
source quantum_package.rc
if command -v qp_run >/dev/null 2>&1; then
    echo "Quantum Package installation successful!"
    echo "Available modules:"
    ls src/ | head -5
else
    echo "Warning: qp_run not found in PATH, but installation may still be functional"
fi

echo "Quantum Package installation completed!"
echo "Please run 'source ~/.bashrc' to load the environment variables."