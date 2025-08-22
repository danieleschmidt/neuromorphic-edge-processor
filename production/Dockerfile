# Production Dockerfile for Neuromorphic Edge Processor
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies  
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY setup.py .
COPY production_config.json .

# Install the package
RUN pip install -e .

# Create non-root user
RUN useradd -m -s /bin/bash neuromorphic
RUN chown -R neuromorphic:neuromorphic /app
USER neuromorphic

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "from src.models.lif_neuron import LIFNeuron; print('healthy')" || exit 1

# Run application
CMD ["python", "-m", "src.models.lif_neuron"]
