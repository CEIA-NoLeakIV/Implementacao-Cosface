# Landmark-Conditioned Face Recognition (CosFace Refactor)

Este repositório contém uma implementação de **Reconhecimento Facial Condicionado por Landmarks**, utilizando uma arquitetura de dois ramos para fundir características visuais (imagem) com características geométricas (pontos faciais). O projeto foi refatorado para utilizar o framework **CosFace** como função de perda e integra a biblioteca **UniFace** para detecção robusta de faces e extração de landmarks.

## 🧠 Arquitetura do Modelo

A rede neural utiliza uma abordagem de fusão tardia de características (*late fusion*), composta por dois ramos principais:

1.  **Ramo Visual (Backbone):**
    * **Entrada:** Imagem facial alinhada (112x112 RGB).
    * **Modelo:** ResNet50 (pré-treinada na ImageNet) ou MobileNetV3.
    * **Saída:** Vetor de *embedding* visual (512 dimensões).

2.  **Ramo Geométrico (Landmark Encoder):**
    * **Entrada:** Coordenadas normalizadas de 5 landmarks faciais (olho esquerdo, olho direito, nariz, boca esquerda, boca direita).
    * **Modelo:** MLP (Multi-Layer Perceptron) com camadas Lineares, BatchNorm e PReLU.
    * **Saída:** Vetor de *embedding* geométrico (128 dimensões).

3.  **Módulo de Fusão:**
    * Concatena os vetores visual (512d) e geométrico (128d).
    * Passa por uma camada densa para projetar o resultado final num espaço de 512 dimensões.

**Função de Perda:**
* Utiliza **Margin Cosine Product (MCP/CosFace)** para maximizar a separação inter-classes e minimizar a variação intra-classe.

---

## 🛠️ Requisitos e Instalação

O projeto requer um ambiente Python 3.10+ e bibliotecas específicas para evitar conflitos de GPU entre PyTorch e ONNX Runtime.

### Dependências Principais
* PyTorch >= 2.0 (com suporte CUDA)
* UniFace >= 1.1.2 (para detecção via SCRFD/RetinaFace)
* ONNX Runtime GPU

### Instalação

```bash
# 1. Clone o repositório
git clone <url-do-repositorio>
cd Cosface_Refactor

# 2. Instale as dependências (ordem recomendada para evitar conflitos)
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
pip install onnxruntime-gpu
pip install uniface==1.1.2
pip install -r requirements.txt
```
## 🚀 Como Usar

Devido a conflitos conhecidos entre os drivers CUDA carregados pelo PyTorch e pelo ONNX Runtime, o processo de treinamento é dividido em duas etapas: Preparação e Treinamento.
   
   **Passo 1: Preparação dos Dados (Extração de Landmarks)**

Este script isolado utiliza a GPU exclusivamente para o UniFace detectar faces e extrair landmarks, salvando um cache JSON. Isso evita gargalos de CPU durante o treino.
Bash

```bash
python prepare_data.py \
    --root /caminho/para/dataset/train \
    --dataset-fraction 1.0  # Use 0.3 para testes rápidos com 30% dos dados
```

Isso gerará um arquivo landmark_cache/<dataset>_landmarks.json.
   
   **Passo 2: Treinamento do Modelo**

O script de treino carrega o cache gerado e treina a rede neural.

```bash
python train.py \
    --root /caminho/para/dataset/train \
    --database VggFace2 \
    --network resnet50 \
    --classifier MCP \
    --use-landmarks \
    --landmark-cache-dir landmark_cache \
    --epochs 30 \
    --batch-size 32 \
    --lr 0.05 \
    --save-path weights/resnet50_run1
```

   **Argumentos Importantes (train.py)**

```
Argumento	Descrição
--use-landmarks	Ativa a arquitetura de dois ramos (Visual + Landmarks).
--network	Define o backbone visual (ex: resnet50, mobilenetv3_large).
--dataset-fraction	Define a % do dataset a ser usada (deve corresponder ao cache gerado).
--lr	Taxa de aprendizado inicial (recomendado 0.05 a 0.001 dependendo do batch).
```

## 📁 Estrutura do Projeto

```
Cosface_Refactor/
├── models/
│   ├── landmark_conditioned.py  # Arquitetura de fusão (Encoder + Fusion)
│   ├── resnet.py                # Backbone ResNet50 customizado
│   └── ...
├── utils/
│   ├── landmark_annotator.py    # Wrapper robusto para o UniFace/SCRFD
│   ├── dataset.py               # ImageFolder modificado para carregar landmarks
│   └── metrics.py               # Implementação do CosFace (MCP)
├── prepare_data.py              # Script de pré-processamento isolado
├── train.py                     # Loop principal de treinamento
└── requirements.txt
```
