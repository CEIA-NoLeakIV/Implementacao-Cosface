# Landmark-Conditioned Face Recognition Framework

Este repositório contém uma implementação personalizada de um framework de reconhecimento facial que utiliza uma arquitetura de **dois ramos (Two-Branch Architecture)**: um ramo visual (backbone CNN) e um ramo geométrico (Landmarks), fundidos para gerar um embedding final mais robusto.

O projeto foi refatorado para resolver conflitos de drivers entre PyTorch e ONNX Runtime, utilizando uma estratégia de execução em duas etapas.

## 🧠 Arquitetura do Modelo

O modelo `LandmarkConditionedModel` combina informações visuais e geométricas:

1.  **Ramo Visual (Backbone):**
    * Utiliza **ResNet50** (pré-treinada na ImageNet) ou outras arquiteturas (MobileNet, SphereFace).
    * Entrada: Imagem RGB (112x112).
    * Saída: Embedding Visual (512d).

2.  **Ramo de Landmarks:**
    * Utiliza um **Encoder MLP** (Multi-Layer Perceptron) personalizado.
    * Entrada: Coordenadas normalizadas (x, y) de 5 pontos faciais extraídos pelo **Uniface (RetinaFace/SCRFD)**.
    * Saída: Embedding Geométrico (128d).

3.  **Fusão (Feature Fusion):**
    * Concatena os vetores visual e geométrico.
    * Passa por camadas lineares e de normalização (BatchNorm1d + PReLU) para projetar no espaço final de 512 dimensões.

## 🛠️ Pré-requisitos e Instalação

O projeto requer um ambiente com suporte a GPU e bibliotecas específicas para evitar conflitos de versão.

**Dependências Principais:**
* Python 3.10+
* PyTorch (com suporte a CUDA)
* `uniface` (Versão 1.1.2 ou superior)
* `onnxruntime-gpu`

**Instalação:**

```bash
# 1. Instalar dependências básicas
pip install -r requirements.txt

# 2. Instalar versão específica do Uniface (Crítico para compatibilidade de retorno)
pip install uniface==1.1.2

# 3. Garantir ONNX Runtime GPU (para extração rápida de landmarks)
pip install onnxruntime-gpu

bash```


🚀 Como Usar

Devido a conflitos de alocação de memória e drivers CUDA entre o PyTorch (treino) e o ONNX Runtime (detecção de faces), o processo foi dividido em dois scripts sequenciais.
Passo 1: Preparação de Dados (Extração de Landmarks)

Este script roda isolado, sem carregar o PyTorch, permitindo que o uniface use a GPU livremente para detectar faces e extrair landmarks.
Bash

python prepare_data.py \
    --root path/to/dataset/train \
    --dataset-fraction 0.3 \
    --cache-dir landmark_cache

    --dataset-fraction: Define a porcentagem do dataset a ser processada (ex: 0.3 para 30%). Útil para Sanity Checks rápidos.

    Saída: Gera um arquivo JSON em landmark_cache/ contendo as coordenadas normalizadas.

Passo 2: Treinamento

O script de treino carrega o cache gerado e inicia o treinamento da rede neural.
Bash

python train.py \
    --root path/to/dataset/train \
    --database VggFace2 \
    --network resnet50 \
    --classifier MCP \
    --use-landmarks \
    --landmark-cache-dir landmark_cache \
    --dataset-fraction 0.3 \
    --epochs 25 \
    --batch-size 32 \
    --lr 0.001 \
    --save-path weights/resnet50_landmark

Argumentos Importantes:

    --use-landmarks: Ativa a arquitetura de dois ramos e o carregamento do JSON.

    --dataset-fraction: Deve corresponder à fração usada na preparação.

    --classifier: Função de perda (ex: MCP para Margin Cosine Product / CosFace).

    --lr: Taxa de aprendizado (Recomendado 0.001 para ResNet50 pré-treinada).

📊 Estrutura de Arquivos

    models/landmark_conditioned.py: Definição da arquitetura de fusão e encoders.

    utils/landmark_annotator.py: Lógica robusta de extração usando Uniface v1.1.2 com fallback de erros.

    prepare_data.py: Script isolado para geração de cache de landmarks.

    train.py: Script principal de treinamento com suporte a argumentos de landmarks.

📝 Notas sobre Resultados

    Loss Function: O uso de CosFace (MCP) com margem 0.40 exige um ajuste fino do Learning Rate.

    Comportamento Inicial: É esperado que a acurácia comece baixa e a Loss alta (~20+) nas primeiras épocas devido ao "Cold Start" da camada de fusão, que é inicializada aleatoriamente e precisa se alinhar com o backbone pré-treinado.
