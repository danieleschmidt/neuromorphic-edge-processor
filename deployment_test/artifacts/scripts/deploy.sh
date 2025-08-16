#!/bin/bash
set -e

echo "Deploying neuromorphic-edge-processor..."

# Build and deploy based on target
case "local" in
    "docker")
        docker-compose up -d
        ;;
    "kubernetes")
        kubectl apply -f kubernetes/
        ;;
    "edge_device")
        sudo systemctl enable neuromorphic-edge-processor.service
        sudo systemctl start neuromorphic-edge-processor.service
        ;;
    *)
        echo "Unknown target: local"
        exit 1
        ;;
esac

echo "Deployment completed"