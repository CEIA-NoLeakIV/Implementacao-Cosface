
# Dockerfile adaptado para CosFace Framework
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Configurar ambiente
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Instalar Python e dependências básicas
RUN apt-get update && apt-get install -y \
	python3.10 \
	python3-pip \
	git \
	wget \
	libgl1-mesa-glx \
	libglib2.0-0 \
	&& rm -rf /var/lib/apt/lists/*

# Criar symlink para python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# Atualizar pip
RUN pip install --upgrade pip

# Diretório de trabalho
WORKDIR /workspace/cosface

# Copiar requirements e instalar dependências
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código do projeto
COPY . .

# Criar diretórios
RUN mkdir -p data weights logs

# Comando padrão - shell interativo
CMD ["/bin/bash"]
[]