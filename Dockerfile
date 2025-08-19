# Neuromorphic Edge Processor - Production Dockerfile
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
COPY setup.py .
COPY README.md .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim as production

# Create non-root user
RUN groupadd -r neuromorphic && useradd -r -g neuromorphic neuromorphic

# Set working directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ ./src/
COPY examples/ ./examples/
COPY benchmarks/ ./benchmarks/
COPY setup.py ./
COPY requirements.txt ./
COPY README.md ./

# Install the package
RUN pip install -e .

# Create necessary directories
RUN mkdir -p /app/outputs /app/logs /app/data && \
    chown -R neuromorphic:neuromorphic /app

# Switch to non-root user
USER neuromorphic

# Set environment variables
ENV PYTHONPATH=/app/src
ENV NEUROMORPHIC_LOG_LEVEL=INFO
ENV NEUROMORPHIC_OUTPUT_DIR=/app/outputs
ENV NEUROMORPHIC_DATA_DIR=/app/data

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import src.models.spiking_neural_network; print('Healthy')" || exit 1

# Expose port for potential web interface
EXPOSE 8080

# Default command
CMD ["python", "-m", "benchmarks.cli", "--help"]

# Labels for metadata
LABEL name="neuromorphic-edge-processor"
LABEL version="0.1.0"
LABEL description="Brain-inspired ultra-low power computing at the edge"
LABEL maintainer="Daniel Schmidt <daniel@terragonlabs.com>"