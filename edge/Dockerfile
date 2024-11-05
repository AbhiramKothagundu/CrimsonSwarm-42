# Dockerfile for Edge Node
FROM python:3.8-slim

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && pip install opencv-python-headless requests flask psutil datetime

# Copy the edge camera script into the container
COPY edge_camera.py /app/edge_camera.py
COPY templates /app/templates

# Set the working directory
WORKDIR /app

# Run the edge camera script
CMD ["python", "edge_camera.py"]
