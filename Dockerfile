# Base image com suporte a GPUs Nvidia e Python 3.11
FROM nvidia/cuda:12.0.1-cudnn8-runtime-ubuntu20.04

# Definir variáveis de ambiente
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TF_ENABLE_ONEDNN_OPTS=0

# Atualizar pacotes e instalar dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    zlib1g-dev \
    libffi-dev \
    libssl-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    curl \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    liblzma-dev \
    git zip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Baixar e compilar o Python 3.11
RUN wget https://www.python.org/ftp/python/3.11.0/Python-3.11.0.tgz && \
    tar xvf Python-3.11.0.tgz && \
    cd Python-3.11.0 && \
    ./configure --enable-optimizations && \
    make -j$(nproc) && \
    make altinstall && \
    cd .. && rm -rf Python-3.11.0 Python-3.11.0.tgz

# Criar link simbólico para o comando 'python'
RUN ln -s /usr/local/bin/python3.11 /usr/bin/python

# Definir o diretório de trabalho
WORKDIR /app

# Copiar os arquivos do projeto para o container
COPY . /app

# Instalar dependências do Python
RUN python3.11 -m pip install --upgrade pip && \
RUN python3.11 -m pip install --upgrade pip && \
    python3.11 -m pip install tensorflow==2.17.0 && \
    python3.11 -m pip install -r requirements.txt --no-deps

# Comando padrão ao iniciar o container
CMD ["bash"]