````markdown
# Face Recognition System: CosFace + Milvus Standalone

Este repositório contém uma plataforma modular e de alta performance para reconhecimento facial, integrando o backbone **ResNet50** (otimizado com **CosFace Loss**) ao banco de dados vetorial **Milvus Standalone**.

O projeto foi evoluído de scripts lineares para uma arquitetura modular, permitindo testes de escalabilidade, benchmarking de hardware e buscas vetoriais sub-milissegundos com metadados.

---

## 🏗️ Arquitetura Técnica

- **Modelo:** ResNet50 com função de perda **CosFace**, treinada para gerar embeddings altamente discriminativos.
- **Extração (TTA):** Implementação de **Test Time Augmentation (TTA)**, concatenando o vetor da imagem original com a versão espelhada para gerar um embedding final de **1024 dimensões**.
- **Persistência Vetorial:** **Milvus Standalone (v2.4.0)** via Docker.  
  - **Diferença Crucial:** Diferente do Milvus Lite, a versão Standalone utilizada aqui suporta indexação **HNSW (Hierarchical Navigable Small World)** massiva e persistência robusta em disco, essencial para aplicações de produção.

---

## 📂 Estrutura do Repositório

A estrutura foi organizada de forma modular para facilitar a manutenção e a integração com APIs:

- **`models/`**  
  Definições das arquiteturas neurais (backbones).

- **`tools/`**  
  Scripts utilitários para métricas, gerenciamento de datasets e monitoramento de hardware.

- **`recursos/`**  
  Diretório centralizado para dados de suporte, incluindo:
  - imagens de teste  
  - embeddings pré-calculados (`.npy`)  
  - mapeamento de nomes (`.json`)

- **`pesos/`**  
  Armazenamento de checkpoints do modelo (`.ckpt`).

- **`Notebooks/`**  
  Ambiente interativo para experimentação e benchmarking modular.

---

## 🛠️ Configuração do Ambiente

### 1. Subir a Infraestrutura (Docker)

O sistema utiliza o **Docker Compose** para gerenciar os serviços:

- **Milvus** (banco vetorial)
- **MinIO** (armazenamento)
- **Etcd** (metadados)

A conexão gRPC local é mapeada para a porta **19540** no host.

```bash
docker-compose up -d
````

---

### 2. Instalação de Dependências

Instale as bibliotecas Python necessárias para treinamento, inferência e comunicação com o Milvus:

* PyMilvus
* PyTorch
* NumPy
* Psutil

```bash
pip install -r requirements.txt
```

---

## 📓 Guia de Uso (Notebooks)

O fluxo de trabalho é dividido em **dois estágios independentes**, permitindo ingestão, benchmarking, indexação e busca de forma desacoplada.

---

### 📘 Notebook 1: Ingestão e Benchmarking

[Notebook 1: Ingestão e Benchmarking de Performance.ipynb](https://www.google.com/search?q=Notebook%25201%253A%2520Ingest%25C3%25A3o%2520e%2520Benchmarking%2520de%2520Performance.ipynb)

Funcionalidades principais:

* Estabelece a conexão gRPC com o servidor Milvus local.
* Cria a coleção vetorial de forma modular, incluindo campos de metadados:

  * `person_id`: identificador único da face
  * `image_path`: rastreabilidade da imagem de origem
* Executa benchmarking comparativo entre:

  * **Inserção Individual**
  * **Inserção em Lote (Bulk)**

Objetivo: avaliar desempenho de ingestão sob diferentes estratégias e cargas.

---

### 📙 Notebook 2: Indexação e Busca

[Notebook 2: Indexação e Busca.ipynb](https://www.google.com/search?q=Notebook%25202%253A%2520Indexa%25C3%25A7%25C3%25A3o%2520e%2520Busca.ipynb)

Funcionalidades principais:

* Configuração do índice **HNSW (Hierarchical Navigable Small World)** para buscas ANN.
* Ajuste de parâmetros de indexação e busca para otimização de latência.
* Execução de buscas por **similaridade de cosseno**.
* Recuperação de metadados completos associados aos vetores retornados.

---

## 🚀 Scripts Principais

* **`train_resnet_tuned.py`**
  Fine-tuning da arquitetura ResNet50 utilizando **CosFace Loss**.

* **`inference.py`**
  Extração de embeddings faciais a partir de imagens brutas.

* **`milvus_benchmark.py`**
  Pipeline automatizado de benchmark para ingestão e performance.

* **`milvus_search.py`**
  Implementação da lógica de identificação facial em tempo real.
