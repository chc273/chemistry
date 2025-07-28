#!/bin/bash
# Quantum Chemistry Docker Management Script
# Provides easy commands for different Docker environments

set -e

# Default configuration
PROJECT_NAME="quantum-chemistry"
BASE_COMPOSE="docker-compose.external.yml"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_help() {
    echo "Quantum Chemistry Docker Management"
    echo
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo
    echo "Commands:"
    echo "  dev       Start development environment with Jupyter Lab"
    echo "  prod      Start production environment"
    echo "  test      Run test suite"
    echo "  build     Build all containers"
    echo "  clean     Clean up containers and volumes"
    echo "  logs      Show logs for services"
    echo "  shell     Open shell in development container"
    echo "  jupyter   Open Jupyter Lab (development only)"
    echo "  benchmark Run performance benchmarks"
    echo "  lint      Run code quality checks"
    echo
    echo "Environment-specific commands:"
    echo "  dev-up    Start development services"
    echo "  dev-down  Stop development services"
    echo "  prod-up   Start production services" 
    echo "  prod-down Stop production services"
    echo
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  -v, --verbose  Verbose output"
    echo "  --no-cache     Build without cache"
    echo
    echo "Examples:"
    echo "  $0 dev                    # Start development environment"
    echo "  $0 test                   # Run tests"
    echo "  $0 build --no-cache       # Rebuild all containers"
    echo "  $0 shell                  # Open development shell"
    echo "  $0 logs quantum-chemistry-dev  # Show dev container logs"
}

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

# Check Docker availability
check_docker() {
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed or not in PATH"
    fi
    
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running"
    fi
    
    # Check for Docker Compose (either docker-compose or docker compose)
    if command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
    elif docker compose version &> /dev/null; then
        COMPOSE_CMD="docker compose"
    else
        error "Docker Compose is not installed or not in PATH"
    fi
}

# Build containers
build_containers() {
    local no_cache=""
    if [[ "$1" == "--no-cache" ]]; then
        no_cache="--no-cache"
    fi
    
    log "Building quantum chemistry containers..."
    $COMPOSE_CMD -f $BASE_COMPOSE build $no_cache
    
    log "Building combined development image..."
    $COMPOSE_CMD -f $BASE_COMPOSE -f docker-compose.dev.yml build $no_cache quantum-chemistry-dev
    
    log "Building production image..."
    $COMPOSE_CMD -f $BASE_COMPOSE -f docker-compose.prod.yml build $no_cache quantum-chemistry-prod
    
    log "Build completed successfully!"
}

# Development environment
start_dev() {
    log "Starting development environment..."
    $COMPOSE_CMD -f $BASE_COMPOSE -f docker-compose.dev.yml up -d
    
    echo
    echo -e "${BLUE}Development environment started!${NC}"
    echo -e "${BLUE}Jupyter Lab: http://localhost:8888${NC} (token: quantum-dev)"
    echo -e "${BLUE}PostgreSQL: localhost:5432${NC} (user: qc_dev, password: dev_password)"
    echo -e "${BLUE}Redis: localhost:6379${NC}"
    echo
    echo "Use '$0 shell' to open a development shell"
    echo "Use '$0 logs' to view logs"
    echo "Use '$0 dev-down' to stop services"
}

stop_dev() {
    log "Stopping development environment..."
    $COMPOSE_CMD -f $BASE_COMPOSE -f docker-compose.dev.yml down
    log "Development environment stopped."
}

# Production environment
start_prod() {
    if [[ ! -f ".env" ]]; then
        warn "No .env file found. Creating template..."
        cat > .env << EOF
DB_PASSWORD=change_this_password
ENVIRONMENT=production
EOF
        warn "Please edit .env file with production values before starting"
        return 1
    fi
    
    log "Starting production environment..."
    $COMPOSE_CMD -f $BASE_COMPOSE -f docker-compose.prod.yml up -d
    
    echo
    echo -e "${BLUE}Production environment started!${NC}"
    echo -e "${BLUE}Web interface: http://localhost${NC}"
    echo
    echo "Use '$0 logs' to view logs"
    echo "Use '$0 prod-down' to stop services"
}

stop_prod() {
    log "Stopping production environment..."
    $COMPOSE_CMD -f $BASE_COMPOSE -f docker-compose.prod.yml down
    log "Production environment stopped."
}

# Testing
run_tests() {
    log "Running test suite..."
    $COMPOSE_CMD -f $BASE_COMPOSE -f docker-compose.test.yml run --rm test
    log "Tests completed!"
}

run_benchmarks() {
    log "Running performance benchmarks..."
    $COMPOSE_CMD -f $BASE_COMPOSE -f docker-compose.test.yml run --rm benchmark
    log "Benchmarks completed!"
}

run_lint() {
    log "Running code quality checks..."
    $COMPOSE_CMD -f $BASE_COMPOSE -f docker-compose.test.yml run --rm lint
    log "Code quality checks completed!"
}

# Utility functions
open_shell() {
    log "Opening development shell..."
    $COMPOSE_CMD -f $BASE_COMPOSE -f docker-compose.dev.yml exec quantum-chemistry-dev bash
}

show_logs() {
    local service=$1
    if [[ -z "$service" ]]; then
        $COMPOSE_CMD -f $BASE_COMPOSE -f docker-compose.dev.yml logs -f
    else
        $COMPOSE_CMD -f $BASE_COMPOSE -f docker-compose.dev.yml logs -f $service
    fi
}

clean_up() {
    log "Cleaning up containers and volumes..."
    $COMPOSE_CMD -f $BASE_COMPOSE -f docker-compose.dev.yml down -v --remove-orphans
    $COMPOSE_CMD -f $BASE_COMPOSE -f docker-compose.prod.yml down -v --remove-orphans
    $COMPOSE_CMD -f $BASE_COMPOSE -f docker-compose.test.yml down -v --remove-orphans
    
    # Remove unused images
    docker image prune -f
    
    log "Cleanup completed!"
}

# Main command processing
main() {
    check_docker
    
    case "${1:-help}" in
        "help"|"-h"|"--help")
            print_help
            ;;
        "build")
            shift
            build_containers "$@"
            ;;
        "dev")
            start_dev
            ;;
        "dev-up")
            start_dev
            ;;
        "dev-down")
            stop_dev
            ;;
        "prod")
            start_prod
            ;;
        "prod-up")
            start_prod
            ;;
        "prod-down")
            stop_prod
            ;;
        "test")
            run_tests
            ;;
        "benchmark")
            run_benchmarks
            ;;
        "lint")
            run_lint
            ;;
        "shell")
            open_shell
            ;;
        "jupyter")
            echo -e "${BLUE}Jupyter Lab: http://localhost:8888${NC} (token: quantum-dev)"
            ;;
        "logs")
            shift
            show_logs "$@"
            ;;
        "clean")
            clean_up
            ;;
        *)
            error "Unknown command: $1. Use '$0 help' for usage information."
            ;;
    esac
}

# Change to the docker directory
cd "$(dirname "$0")"

# Run main function
main "$@"