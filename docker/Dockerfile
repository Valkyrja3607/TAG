FROM nvidia/cuda:11.7.1-devel-ubuntu22.04

RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

ENV PYTHON_VERSION 3.11.4
ENV PYENV_ROOT /root/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        git \
        curl \
        build-essential \
        libffi-dev \
        libssl-dev \
        zlib1g-dev \
        liblzma-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        llvm \
        libncursesw5-dev \
        xz-utils \
        tk-dev \
        libxml2-dev \
        libxmlsec1-dev \
        libffi-dev \
        liblzma-dev \
        libgl1-mesa-dev

RUN curl https://pyenv.run | bash

RUN pyenv install $PYTHON_VERSION
RUN pyenv global $PYTHON_VERSION

RUN pip install --upgrade pip && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

COPY ./requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
RUN pip install --no-build-isolation git+https://github.com/lucasb-eyer/pydensecrf.git

ENV PYTHONPATH "/workspace/tag:/workspace/tag/scripts:/workspace/tag/scripts/retrieve_utils:/workspace/tag/scripts/retrieve_utils/src"
