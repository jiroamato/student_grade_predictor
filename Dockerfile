FROM quay.io/jupyter/minimal-notebook:afe30f0c9ad8

COPY conda-linux-64.lock /tmp/conda-linux-64.lock

# Install packages, clean cache, and fix permissions in a single layer
RUN conda update --quiet --file /tmp/conda-linux-64.lock \
    && pip install python-json-logger==2.0.7 altair_ally==0.1.1 deepchecks==0.19.1 pandera==0.27.0 \
    && conda clean --all -y -f \
    && fix-permissions "${CONDA_DIR}" \
    && fix-permissions "/home/${NB_USER}"