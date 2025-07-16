# Use lightweight Python base
FROM python:3.10-slim

# Prevent prompts and set safer env defaults
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies needed for OpenCV, pyzbar, etc.
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libzbar0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files into container
COPY . /app

# Install Python packages
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose port for Streamlit
EXPOSE 10000

# Start the app using Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=10000", "--server.address=0.0.0.0"]
