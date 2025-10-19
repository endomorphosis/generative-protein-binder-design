FROM python:3.11

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    vim \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Jupyter and related packages
RUN pip install --no-cache-dir \
    jupyter \
    jupyterlab \
    notebook \
    ipykernel \
    matplotlib \
    numpy \
    pandas \
    requests \
    biopython \
    py3Dmol

# Create a non-root user
RUN useradd -m -u 1000 scientist && \
    chown -R scientist:scientist /workspace

# Copy the notebook
COPY --chown=scientist:scientist ../src/protein-binder-design.ipynb /workspace/

USER scientist

# Expose Jupyter port
EXPOSE 8888

# Configure Jupyter
RUN jupyter notebook --generate-config && \
    echo "c.NotebookApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.open_browser = False" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.token = ''" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.password = ''" >> ~/.jupyter/jupyter_notebook_config.py

# Start Jupyter
CMD ["jupyter", "notebook", "--no-browser", "--ip=0.0.0.0", "--port=8888"]
