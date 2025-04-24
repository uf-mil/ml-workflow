FROM python:3.8-slim

# Install system packages for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy application code
COPY . .

# Copy and install dependencies
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

EXPOSE 2704

# Start Flask app
CMD ["python", "src/app.py"]