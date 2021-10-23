# Base Image
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9-slim

# Setup working directory
WORKDIR /app

# Expose port
EXPOSE 80

# Copy env file
COPY requirements.txt .

# Set up env
RUN ["pip", "install", "-r","./requirements.txt"]

# Copy necessary files
COPY . .

# Start service
CMD ["python3","main.py"]


