FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt

# Copy application code
COPY server.py .
COPY model_backends.py .

# Expose port
EXPOSE 8000

# Run the server
CMD ["python", "server.py"]
