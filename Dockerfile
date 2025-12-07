# use the miniforge base, make sure you specify a verion
FROM condaforge/miniforge3:latest

# copy the lockfile into the container
COPY conda-lock.yml conda-lock.yml

# setup conda-lock and install packages from lockfile
RUN conda install -n base -c conda-forge conda-lock jupyterlab nb_conda_kernels -y
RUN conda-lock install -n 522-milestone conda-lock.yml

# --- NEW: Install Quarto for ARM64 Manually ---
# 1. Install curl (needed to download the file)
RUN apt-get update && apt-get install -y curl

# 2. Download and install the specific ARM64 .deb file
# (We use version 1.8.26 here, but you can update the URL to a newer version later)
RUN curl -LO https://github.com/quarto-dev/quarto-cli/releases/download/v1.8.26/quarto-1.8.26-linux-arm64.deb \
    && dpkg -i quarto-1.8.26-linux-arm64.deb \
    && rm quarto-1.8.26-linux-arm64.deb

# expose JupyterLab port
EXPOSE 8888

# sets the default working directory
# this is also specified in the compose file
WORKDIR /workspace

# Append the hook to .bashrc so every new Jupyter terminal gets it automatically
RUN echo 'eval "$(/opt/conda/bin/conda shell.bash hook)"' >> ~/.bashrc

# run JupyterLab on container start
# uses the jupyterlab from the install environment
CMD ["conda", "run", "--no-capture-output", "-n", "522-milestone", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--IdentityProvider.token=''", "--ServerApp.password=''"]