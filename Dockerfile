# Base image
FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel

# Set environment variable to ensure output is not buffered
# ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /workspace

# Copy requirements.txt to the working directory
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies if needed
RUN apt-get update && apt-get install -y \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Optional: Copy your project files into the container
COPY . /workspace

# Default command to run when starting the container
CMD ["bash"]
