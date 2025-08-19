#!/bin/bash
# Neuromorphic Edge Processor - Production Deployment Script

set -e

# Configuration
PROJECT_NAME="neuromorphic-edge-processor"
IMAGE_NAME="neuromorphic-edge-processor"
REGISTRY=${REGISTRY:-"ghcr.io/danieleschmidt"}
VERSION=${VERSION:-"latest"}
ENVIRONMENT=${ENVIRONMENT:-"production"}
COMPOSE_FILE=${COMPOSE_FILE:-"docker-compose.yml"}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check dependencies
check_dependencies() {
    log_info "Checking deployment dependencies..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    log_success "All dependencies are available"
}

# Pre-deployment validation
validate_environment() {
    log_info "Validating environment configuration..."
    
    # Check if required directories exist
    mkdir -p data outputs logs
    
    # Set proper permissions
    chmod 755 data outputs logs
    
    # Validate configuration files
    if [[ ! -f "$COMPOSE_FILE" ]]; then
        log_error "Docker Compose file not found: $COMPOSE_FILE"
        exit 1
    fi
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    log_success "Environment validation passed"
}

# Build Docker images
build_images() {
    log_info "Building Docker images..."
    
    # Build production image
    docker build \
        --target production \
        --tag "$REGISTRY/$IMAGE_NAME:$VERSION" \
        --tag "$REGISTRY/$IMAGE_NAME:latest" \
        --label "version=$VERSION" \
        --label "environment=$ENVIRONMENT" \
        --label "build-date=$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        .
    
    log_success "Docker images built successfully"
}

# Run quality gates
run_quality_gates() {
    log_info "Running quality gates..."
    
    # Run security scan (if available)
    if command -v trivy &> /dev/null; then
        log_info "Running security scan with Trivy..."
        trivy image --severity HIGH,CRITICAL "$REGISTRY/$IMAGE_NAME:$VERSION"
    else
        log_warning "Trivy not available, skipping security scan"
    fi
    
    # Run functionality tests
    log_info "Running functionality tests..."
    docker run --rm \
        -v "$(pwd)/outputs:/app/outputs" \
        "$REGISTRY/$IMAGE_NAME:$VERSION" \
        python -m pytest tests/test_core_functionality.py -v
    
    log_success "Quality gates passed"
}

# Deploy services
deploy_services() {
    log_info "Deploying services with Docker Compose..."
    
    # Set environment variables for compose
    export NEUROMORPHIC_VERSION="$VERSION"
    export NEUROMORPHIC_REGISTRY="$REGISTRY"
    export NEUROMORPHIC_ENVIRONMENT="$ENVIRONMENT"
    
    # Stop existing services
    docker-compose -f "$COMPOSE_FILE" down --remove-orphans
    
    # Start services
    docker-compose -f "$COMPOSE_FILE" up -d
    
    # Wait for health checks
    log_info "Waiting for services to be healthy..."
    sleep 30
    
    # Check service health
    for service in $(docker-compose -f "$COMPOSE_FILE" config --services); do
        if docker-compose -f "$COMPOSE_FILE" ps "$service" | grep -q "Up (healthy)"; then
            log_success "Service $service is healthy"
        else
            log_warning "Service $service may not be healthy"
        fi
    done
    
    log_success "Services deployed successfully"
}

# Post-deployment validation
post_deployment_validation() {
    log_info "Running post-deployment validation..."
    
    # Check service availability
    if docker-compose -f "$COMPOSE_FILE" ps | grep -q "Up"; then
        log_success "Services are running"
    else
        log_error "Some services failed to start"
        docker-compose -f "$COMPOSE_FILE" logs
        exit 1
    fi
    
    # Run basic functionality test
    log_info "Testing basic functionality..."
    docker-compose -f "$COMPOSE_FILE" exec -T neuromorphic-api \
        python -c "
import sys
sys.path.append('/app/src')
from models.spiking_neural_network import SpikingNeuralNetwork
model = SpikingNeuralNetwork([10, 5, 2])
print('✓ Neuromorphic models load successfully')
"
    
    log_success "Post-deployment validation passed"
}

# Cleanup function
cleanup() {
    log_info "Performing cleanup..."
    
    # Remove dangling images
    docker image prune -f
    
    log_success "Cleanup completed"
}

# Rollback function
rollback() {
    log_error "Deployment failed, initiating rollback..."
    
    # Stop current deployment
    docker-compose -f "$COMPOSE_FILE" down --remove-orphans
    
    # Restore previous version if available
    if docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "$REGISTRY/$IMAGE_NAME:previous"; then
        log_info "Restoring previous version..."
        docker tag "$REGISTRY/$IMAGE_NAME:previous" "$REGISTRY/$IMAGE_NAME:$VERSION"
        deploy_services
        log_success "Rollback completed"
    else
        log_warning "No previous version available for rollback"
    fi
}

# Main deployment function
main() {
    log_info "Starting Neuromorphic Edge Processor deployment..."
    log_info "Version: $VERSION"
    log_info "Environment: $ENVIRONMENT"
    log_info "Registry: $REGISTRY"
    
    # Trap errors for rollback
    trap rollback ERR
    
    check_dependencies
    validate_environment
    build_images
    run_quality_gates
    deploy_services
    post_deployment_validation
    cleanup
    
    log_success "Deployment completed successfully!"
    log_info "Services are available at:"
    log_info "  - API: http://localhost:8080"
    log_info "  - Logs: ./logs/"
    log_info "  - Outputs: ./outputs/"
}

# Command line interface
case "${1:-deploy}" in
    "build")
        build_images
        ;;
    "test")
        run_quality_gates
        ;;
    "deploy")
        main
        ;;
    "rollback")
        rollback
        ;;
    "cleanup")
        cleanup
        ;;
    "logs")
        docker-compose -f "$COMPOSE_FILE" logs -f
        ;;
    "status")
        docker-compose -f "$COMPOSE_FILE" ps
        ;;
    "stop")
        docker-compose -f "$COMPOSE_FILE" down
        ;;
    *)
        echo "Usage: $0 {build|test|deploy|rollback|cleanup|logs|status|stop}"
        echo ""
        echo "Commands:"
        echo "  build    - Build Docker images"
        echo "  test     - Run quality gates"
        echo "  deploy   - Full deployment (default)"
        echo "  rollback - Rollback to previous version"
        echo "  cleanup  - Clean up Docker resources"
        echo "  logs     - Show service logs"
        echo "  status   - Show service status"
        echo "  stop     - Stop all services"
        exit 1
        ;;
esac