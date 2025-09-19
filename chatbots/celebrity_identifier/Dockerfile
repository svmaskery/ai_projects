# Python image
FROM python:3.10-slim

# Helpful environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Work directory inside the docker container
WORKDIR /app

# Installing system dependencies 
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copying all contents from local to container
COPY . .

# Run setup.py
RUN pip install --no-cache-dir -e .

# Expose Flask port
EXPOSE 8000

# Run the app
CMD ["python", "app.py"]