FROM quay.io/jupyter/minimal-notebook:afe30f0c9ad8

COPY conda-linux-64.lock /tmp/conda-linux-64.lock

# Install packages, clean cache, and fix permissions in a single layer
RUN conda update --quiet --file /tmp/conda-linux-64.lock

# Clean cache
RUN conda clean --all -y -f

# Fix permissions
RUN fix-permissions "${CONDA_DIR}" \
    && fix-permissions "/home/${NB_USER}"