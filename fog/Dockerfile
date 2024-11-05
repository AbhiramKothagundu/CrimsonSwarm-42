# Dockerfile for Fog Node
FROM python:3.8-slim

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && pip install opencv-python-headless flask requests psutil datetime

# Copy the fog processor script into the container
COPY fog_processor.py /app/fog_processor.py
COPY templates /app/templates

# Set the working directory
WORKDIR /app

# Run the fog processor script
CMD ["python", "fog_processor.py"]
