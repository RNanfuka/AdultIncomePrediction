# use the miniforge base, make sure you specify a verion
FROM condaforge/miniforge3:latest

# copy the lockfile into the container
COPY conda-lock.yml conda-lock.yml

# setup conda-lock and install packages from lockfile
RUN conda install -n base -c conda-forge conda-lock jupyterlab nb_conda_kernels -yml
RUN conda run -n base conda-lock install -n 522-milestone conda-lock.yml

# Install system utilities (Make, Curl, etc.)
RUN apt-get update \
    && apt-get install -y --no-install-recommends make curl \
    && rm -rf /var/lib/apt/lists/*

# Expose ports
EXPOSE 8888 
EXPOSE 8889
# sets the default working directory
# this is also specified in the compose file
WORKDIR /workspace

# Ensure conda is initialized in terminals
RUN echo 'eval "$(/opt/conda/bin/conda shell.bash hook)"' >> ~/.bashrc

# Environment variable to control which service runs
# Options: "jupyter" OR "quarto"
ENV SERVICE=jupyter

RUN conda run -n 522-milestone quarto --version

# Script that chooses which service to run
# (Installed into the container as /entrypoint.sh)
RUN printf '%s\n' \
'#!/bin/bash' \
'set -e' \
'eval "$(/opt/conda/bin/conda shell.bash hook)"' \
'if [ "$SERVICE" = "quarto" ]; then' \
'  echo "🚀 Starting Quarto preview on port 8889..."' \
'  conda run --no-capture-output -n 522-milestone quarto preview --port 8889 --host 0.0.0.0 --no-browser' \
'else' \
'  echo "🚀 Starting JupyterLab on port 8888..."' \
'  conda run --no-capture-output -n 522-milestone jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --IdentityProvider.token= --ServerApp.password=' \
'fi' \
> /entrypoint.sh \
    && chmod +x /entrypoint.sh

# Use the entrypoint
ENTRYPOINT ["/entrypoint.sh"]