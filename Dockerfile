# Use an official Python runtime as a parent image
FROM nvidia/cuda:12.8.0-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3.9 \
    python3.9-dev \
    python3.9-distutils\
    git \
    curl \
    coreutils \
    && rm -rf /var/lib/apt/lists/*


RUN  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
RUN apt-get update && apt-get install -y \
    nvidia-container-toolkit \
    && rm -rf /var/lib/apt/lists/*
# Install pip for Python 3
RUN curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.9 get-pip.py
RUN python3.9 -m pip install setuptools
RUN python3.9 -m pip install nvidia-pyindex
# Set the VLLM_TARGET_DEVICE environment variable
ENV VLLM_TARGET_DEVICE=gpu


ENV PYTHONPATH="/usr/local/lib/python3.9/site-packages"
RUN echo ${PYTHONPATH}

RUN python3.9 -m pip install -r /app/requirements.txt
RUN python3.9 -m pip install flash-attn --no-build-isolation


EXPOSE 8000

COPY . /app

RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]

CMD ["both"]