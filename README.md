# Implementação Cosface + Milvus

Este repositório contém uma implementação do modelo **Cosface** para reconhecimento facial, integrada ao banco de dados vetorial **Milvus** para busca e indexação de embeddings em larga escala.

O projeto abrange desde o ajuste fino (fine-tuning) da arquitetura ResNet até a realização de benchmarks de performance e busca por similaridade.

## 📂 Estrutura do Repositório

A estrutura está organizada para separar a lógica do modelo, ferramentas de processamento e os módulos de integração com o Milvus:

* **`models/`**: Definições das arquiteturas utilizadas (ex: `resnet.py`).
* **`tools/`**: Scripts utilitários para processamento de datasets, métricas de validação, camadas customizadas e acompanhamento de treino.
* **`recursos/`**: Contém arquivos de suporte, como imagens de teste, embeddings pré-calculados (`.npy`) e mapeamento de nomes.
* **`pesos/`**: Diretório destinado ao armazenamento dos pesos dos modelos treinados.
* **`parâmetros/`**: Arquivos de configuração e hiperparâmetros de treino.

## 🚀 Principais Scripts

### Treinamento e Inferência
* **`train_resnet_tuned.py`**: Script principal para realizar o ajuste fino do modelo.
* **`inference.py`**: Realiza a extração de embeddings a partir de imagens de faces.
* **`evaluate_tta_tuned.py`**: Avaliação do modelo utilizando Test-Time Augmentation (TTA).

### Integração com Milvus
* **`milvus_benchmark.py`**: Script para medir a performance de ingestão e busca no Milvus.
* **`milvus_search.py`**: Implementação da lógica de busca vetorial para identificação de faces.

## 📓 Notebooks
Para uma exploração interativa, o repositório inclui:
1.  **`Notebook 1: Ingestão e Benchmarking de Performance.ipynb`**: Focado no fluxo de dados para o banco vetorial e testes de carga.
2.  **`Notebook 2: Indexação e Busca.ipynb`**: Demonstração prática de como realizar buscas por similaridade e gerir índices.

## 🛠️ Instalação

Certifique-se de que tem o Python instalado e execute o comando abaixo para instalar as dependências necessárias:

```bash
pip install -r requirements.txt
