FROM python:3.11-slim

WORKDIR /app

# Tools needed for embedded provisioning (best-effort).
# - rsync: AlphaFold PDB mmCIF download
# - ca-certificates: HTTPS downloads
# - bash/findutils: used by a few best-effort extract steps
RUN apt-get update \
	&& apt-get install -y --no-install-recommends rsync ca-certificates bash findutils gzip tar \
	&& rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt

# Copy application code
COPY *.py .

# Expose port
EXPOSE 8000

# Run the server
CMD ["python", "server.py"]
