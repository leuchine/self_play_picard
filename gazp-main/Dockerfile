FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-devel

ENV LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

RUN mkdir -p /usr/share/man/man1 && \
    apt-get update && apt-get install -y \
    build-essential \
    cifs-utils \
    curl \
    default-jdk \
    dialog \
    dos2unix \
    git \
    sudo

RUN pip install vocab
RUN pip install embeddings
RUN pip install transformers
RUN pip install sqlparse
RUN pip install nltk
RUN pip install loguru