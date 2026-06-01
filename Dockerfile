# Use official Python slim image. 
# (If you require GPU inference later, you can swap this to a pytorch/pytorch:latest-cuda image)
FROM python:3.11-slim

# Set environment variables to prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set the working directory
WORKDIR /app

# Install system dependencies (Required for OpenCV to process images)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install the Python dependencies
# We use --no-cache-dir to keep the Docker image size small
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Expose port 8080 (This is the default port Google Cloud Run looks for)
EXPOSE 8080

# Start the FastAPI server using Uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
