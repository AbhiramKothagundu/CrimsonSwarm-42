# Dockerfile for Cloud Node
FROM python:3.8-slim

# Install Flask
RUN pip install flask requests

# Copy the cloud storage script into the container
COPY cloud_storage.py /app/cloud_storage.py
COPY templates /app/templates

# Set the working directory
WORKDIR /app

# Run the cloud storage script
CMD ["python", "cloud_storage.py"]
