# Quantum Chemistry Docker Environment

This directory contains Docker configurations for the quantum chemistry package, providing development, testing, and production environments with integrated external quantum chemistry methods.

## Quick Start

### Prerequisites
- Docker 20.10+ 
- Docker Compose 2.0+
- 8GB+ RAM (16GB+ recommended for production)

### Start Development Environment
```bash
# Make the run script executable (first time only)
chmod +x run.sh

# Start development environment with Jupyter Lab
./run.sh dev

# Access Jupyter Lab at http://localhost:8888 (token: quantum-dev)
```

### Run Tests
```bash
# Run the complete test suite
./run.sh test

# Run performance benchmarks
./run.sh benchmark

# Run code quality checks
./run.sh lint
```

## Available Environments

### 1. Development Environment
**Files**: `docker-compose.dev.yml`
**Command**: `./run.sh dev`

Features:
- Live code reloading via volume mounts
- Jupyter Lab interface
- PostgreSQL and Redis for development
- All external quantum chemistry methods available
- Debug tools and profiling

**Access Points**:
- Jupyter Lab: http://localhost:8888 (token: quantum-dev)
- PostgreSQL: localhost:5432 (user: qc_dev, password: dev_password)
- Redis: localhost:6379

### 2. Production Environment  
**Files**: `docker-compose.prod.yml`
**Command**: `./run.sh prod`

Features:
- Optimized production builds
- Nginx reverse proxy
- Persistent data volumes
- Health checks and monitoring
- Log aggregation with Fluentd
- Resource limits and scaling

**Requirements**:
- Create `.env` file with production secrets
- Configure SSL certificates in `ssl/` directory
- Set up persistent volumes in `/data/quantum-chemistry/`

### 3. Testing Environment
**Files**: `docker-compose.test.yml`
**Command**: `./run.sh test`

Features:
- Automated test execution
- Code coverage reporting
- Performance benchmarking
- Memory profiling
- Linting and code quality checks

## Docker Images

### Base Images
- `quantum-chemistry/scientific-base`: Ubuntu 24.04 + scientific libraries
- `quantum-chemistry/combined`: Unified image with all external methods

### External Method Images
- `quantum-chemistry/block2`: Block2 DMRG calculations
- `quantum-chemistry/openmolcas`: OpenMolcas CASPT2 calculations  
- `quantum-chemistry/dice`: Dice SHCI calculations
- `quantum-chemistry/quantum-package`: Quantum Package CIPSI calculations

### Application Images
- `quantum-chemistry/dev`: Development image with tools
- `quantum-chemistry/prod`: Production-optimized image
- `quantum-chemistry/test`: Testing image with test frameworks

## External Methods Integration

All external quantum chemistry methods are automatically available in the containers:

```python
# Example usage
from quantum.chemistry.multireference.external import (
    DMRGMethod,      # Block2 DMRG
    AFQMCMethod,     # ipie AF-QMC  
    SelectedCIMethod, # Dice SHCI
)
from quantum.chemistry.multireference.external.openmolcas import CASPT2Method

# Methods automatically detect and use containerized installations
dmrg = DMRGMethod(bond_dimension=1000)
result = dmrg.calculate(scf_obj, active_space)
```

## Management Commands

The `run.sh` script provides convenient commands:

```bash
# Environment management
./run.sh dev        # Start development environment
./run.sh prod       # Start production environment  
./run.sh dev-down   # Stop development environment
./run.sh prod-down  # Stop production environment

# Development tools
./run.sh shell      # Open development shell
./run.sh jupyter    # Show Jupyter Lab URL
./run.sh logs       # Show all logs
./run.sh logs [service]  # Show specific service logs

# Testing and quality
./run.sh test       # Run test suite
./run.sh benchmark  # Run performance benchmarks
./run.sh lint       # Run code quality checks

# Maintenance  
./run.sh build      # Build all containers
./run.sh build --no-cache  # Rebuild without cache
./run.sh clean      # Clean up containers and volumes
```

## Volume Management

### Development Volumes
- Source code: Live-mounted for development
- Cache directories: Persisted for faster rebuilds
- Data directories: Separate volumes for each method

### Production Volumes
- Application data: `/data/quantum-chemistry/data`
- Results: `/data/quantum-chemistry/results`  
- Logs: `/data/quantum-chemistry/logs`
- Database: Managed by PostgreSQL container

## Configuration

### Development Configuration
Configuration is handled through environment variables in the compose files.

### Production Configuration
1. Copy `.env.example` to `.env` and configure:
   ```bash
   DB_PASSWORD=secure_production_password
   ENVIRONMENT=production
   ```

2. Edit `config/production.yml` for application settings

3. Configure SSL certificates in `ssl/` directory for HTTPS

4. Set up persistent volumes:
   ```bash
   sudo mkdir -p /data/quantum-chemistry/{data,results,logs}
   sudo chown -R 1000:1000 /data/quantum-chemistry/
   ```

## Monitoring and Logging

### Health Checks
All production services include health checks:
- Application health via Python imports
- Database connectivity checks
- Redis availability checks

### Logging
- Application logs: JSON format with structured logging
- Access logs: Nginx format for web requests
- Container logs: Available via `docker-compose logs`
- Log aggregation: Fluentd in production environment

### Metrics
- Application metrics: Exposed on port 9090 (Prometheus format)
- Container metrics: Available via Docker stats
- Resource usage: Monitored and limited in production

## Troubleshooting

### Common Issues

**Container build failures**:
```bash
# Clear Docker cache and rebuild
./run.sh clean
./run.sh build --no-cache
```

**Port conflicts**:
```bash
# Check what's using the ports
netstat -tulpn | grep :8888
# Change ports in docker-compose files if needed
```

**Permission errors**:
```bash
# Fix file permissions
sudo chown -R $USER:$USER .
chmod +x run.sh
```

**Out of disk space**:
```bash
# Clean up unused Docker resources
docker system prune -a
./run.sh clean
```

### Performance Tuning

**Memory issues**:
- Increase Docker Desktop memory allocation (8GB minimum)
- Adjust `MOLCAS_MEM` and similar environment variables
- Monitor memory usage: `docker stats`

**CPU optimization**:
- Set `OMP_NUM_THREADS` based on available cores
- Use `docker-compose --parallel` for faster builds
- Enable BuildKit: `export DOCKER_BUILDKIT=1`

### Debugging

**Container debugging**:
```bash
# Open shell in running container
./run.sh shell

# Run container interactively  
docker run -it quantum-chemistry/dev:latest bash

# Check container logs
./run.sh logs quantum-chemistry-dev
```

**Method-specific debugging**:
```bash
# Test individual external methods
docker run --rm quantum-chemistry/block2:latest
docker run --rm quantum-chemistry/dice:latest  
docker run --rm quantum-chemistry/openmolcas:latest
```

## Integration with CI/CD

The testing environment is designed for CI/CD integration:

```yaml
# Example GitHub Actions workflow
- name: Run Quantum Chemistry Tests
  run: |
    cd docker
    ./run.sh build
    ./run.sh test
    ./run.sh benchmark
```

## Security Considerations

### Development Environment
- Uses default passwords (change for any network-accessible deployment)
- Bind mounts source code (keep sensitive data out of source)
- Debug mode enabled (disable in production)

### Production Environment  
- Requires secure password configuration
- Uses secrets management for sensitive data
- Enables SSL/TLS for encrypted communication
- Implements resource limits and network isolation
- Regular security updates via base image updates

For production deployment, follow your organization's security guidelines and consider additional measures like network segmentation, access controls, and security scanning.