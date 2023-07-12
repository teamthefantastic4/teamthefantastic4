FROM python:3.8.13 as base

# Install some packages
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    vim \
    wget \
    curl

# Setup some paths
ENV PYTHONPATH=/home/app/.local/lib/python3.8/site-packages:/home/app/src
ENV PATH=$PATH:/home/app/.local/bin
RUN pip install tensorflow-aarch64 -f https://cf.tf.kmtea.eu/whl/stable.html