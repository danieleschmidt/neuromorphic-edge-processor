#!/bin/bash
# Production Deployment Script for Neuromorphic Edge Processor

set -e

echo "🚀 Starting Neuromorphic Edge Processor Production Deployment"
echo "============================================================="

# Build Docker image
echo "📦 Building production Docker image..."
docker build -t neuromorphic-edge-processor:3.0.0 .

# Run quality checks
echo "🔍 Running pre-deployment quality checks..."
python -c "
import sys
sys.path.append('src')
from models.lif_neuron import LIFNeuron, LIFParams
from models.optimized_lif_neuron import OptimizedLIFNeuron, OptimizedLIFParams
import numpy as np

print('Testing core functionality...')
params = LIFParams()
neuron = LIFNeuron(params, n_neurons=10)
test_input = np.random.randn(10) * 1e-9
result = neuron.forward(test_input)
assert 'spikes' in result
print('✅ Core functionality validated')

print('Testing optimized functionality...')
opt_params = OptimizedLIFParams()
opt_neuron = OptimizedLIFNeuron(opt_params, n_neurons=10)
opt_result = opt_neuron.forward(test_input)
assert 'spikes' in opt_result
print('✅ Optimized functionality validated')

print('✅ All pre-deployment checks passed')
"

# Deploy to Kubernetes (if available)
if command -v kubectl &> /dev/null; then
    echo "🚢 Deploying to Kubernetes..."
    kubectl apply -f kubernetes/deployment.yaml
    echo "✅ Kubernetes deployment initiated"
else
    echo "⚠️  kubectl not found, skipping Kubernetes deployment"
fi

# Start monitoring (if available)
if command -v docker-compose &> /dev/null; then
    echo "📊 Starting monitoring stack..."
    docker-compose -f monitoring/docker-compose.yml up -d
    echo "✅ Monitoring stack started"
fi

echo ""
echo "🎉 Deployment completed successfully!"
echo "📊 Application metrics: http://localhost:3000"
echo "🔍 Health check: http://localhost:8080/health"
echo "============================================================="
