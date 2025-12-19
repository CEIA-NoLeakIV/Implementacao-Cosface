# Estrutura do Framework - Cosface

## Introdução
O framework Cosface foi projetado para treinar e validar modelos de reconhecimento facial utilizando arquiteturas modernas como ResNet50 e perdas específicas como CosFace. Ele é modular e permite flexibilidade na escolha de datasets, arquiteturas e estratégias de treinamento.

---

## Estrutura de Pastas

### Diretórios Principais
- **`config/`**: Contém arquivos de configuração para o treinamento e validação.
  - `face_recognition_config.py`: Define hiperparâmetros e caminhos principais.
  - `training_config.py`: Configurações específicas para o treinamento.

- **`src/`**: Código-fonte principal do framework.
  - **`pipelines/`**: Define os pipelines de treinamento e inferência.
    - `training_pipeline.py`: Gerencia o fluxo de treinamento.
    - `inference_pipeline.py`: Gerencia o fluxo de inferência.
  - **`backbones/`**: Contém implementações de arquiteturas de backbone, como ResNet.
  - **`data_loader/`**: Gerencia o carregamento e pré-processamento de datasets.
    - `face_datasets.py`: Funções para carregar datasets de treino e validação.
  - **`losses/`**: Implementações de funções de perda, como CosFace.
  - **`optimizers/`**: Define otimizadores e agendadores de taxa de aprendizado.

- **`experiments/`**: Resultados de experimentos, incluindo checkpoints, logs e figuras.
  - `checkpoints/`: Modelos salvos durante o treinamento.
  - `logs/`: Logs de treinamento.
  - `figures/`: Gráficos gerados durante o treinamento.

- **`models/`**: Modelos pré-treinados ou salvos.

- **`deliverables/`**: Logs e artefatos antigos (não mais utilizados).

---

## Fluxo de Treinamento e Validação

### Treinamento
1. **Pipeline de Treinamento:**
   - O treinamento é gerenciado pelo arquivo `training_pipeline.py`.
   - O modelo é treinado em duas fases:
     1. **Cabeça do Modelo:** Apenas as camadas finais são treinadas enquanto o backbone está congelado.
     2. **Fine-Tuning Completo:** O backbone é descongelado e o modelo é ajustado em todo o conjunto de dados.

2. **Script de Execução:**
   - O treinamento é iniciado com o script `train.py`.
   - Exemplo de comando:
     ```bash
     python train.py \
     --data /home/ubuntu/noleak/EmbeddingFramework/data/vggface2_HQ_cropado/insightface_aligned_112x112/train
     --epochs 30 \
     ```

### Validação
1. **Pipeline de Validação:**
   - A validação é realizada separadamente após o treinamento, utilizando o script `run_validation.py`.
   - Suporta validação em múltiplos datasets, incluindo LFW.

2. **Script de Execução:**
   - Exemplo de comando:
     ```bash
     python run_validation.py \
     --model /home/ubuntu/noleak/face_embeddings/src/models/Cosface/experiments/finetuning_full_v1/final_model.keras \
     --lfw_root /home/ubuntu/noleak/face_embeddings/data/raw/lfw \
     --pairs /home/ubuntu/noleak/face_embeddings/data/raw/lfw/lfw_ann.txt \
     --save_dir evaluation_final_report \
     ```

---

## Componentes Principais

### Configurações (`config/`)
- Define hiperparâmetros como taxa de aprendizado, número de épocas, e caminhos para datasets e checkpoints.

### Modelos (`src/backbones/`, `src/model_builder/`)
- **`backbones/`:** Contém arquiteturas de redes neurais, como ResNet50.
- **`model_builder/`:** Define como os modelos são construídos, incluindo a adição de camadas específicas como CosFace.

### Datasets (`src/data_loader/`)
- **`face_datasets.py`:**
  - Carrega datasets de treino e validação.
  - Suporta datasets pré-divididos ou divisão automática.

### Otimizadores e Agendadores (`src/optimizers/`)
- Define otimizadores como SGD e Adam.
- Implementa agendadores de taxa de aprendizado, como Cosine Annealing.

### Perdas (`src/losses/`)
- Implementa funções de perda específicas, como CosFace.

---

## Scripts de Execução

### `train.py`
- Inicia o treinamento do modelo.
- Permite especificar o dataset via linha de comando.

### `run_validation.py`
- Realiza a validação do modelo treinado.
- Suporta validação em múltiplos datasets.

### `run_finetuning.py`
- Realiza fine-tuning de modelos pré-treinados em novos datasets.
- Oferece 3 estratégias diferentes de fine-tuning:
  1. **Full Fine-tuning**: Todas as camadas treináveis
  2. **Partial Fine-tuning**: Apenas últimas N camadas
  3. **Differential LR Fine-tuning**: Learning rates diferenciados
- Consulte `FINETUNING.md` para documentação completa.

---

## Como Usar o Framework

1. **Treinar um Modelo:**
   ```bash
   python train.py --dataset_path /path/to/dataset
   ```
```markdown
# Cosface — Framework de Treino, Finetuning e Validação

Este repositório contém um framework modular para reconhecimento facial com a cabeça CosFace (loss) e backbones como ResNet. A documentação abaixo resume como executar treino, finetuning e validação, além de onde os artefatos (checkpoints, logs, figuras) são salvos.

## Sumário rápido
- Treino: `run_training.py` / `src/pipelines/training_pipeline.py`
- Finetuning: `run_finetuning.py` (ver `FINETUNING.md` para detalhes avançados)
- Validação / avaliação: `run_validation.py` e `evaluate.py`
- Configs: `config/face_recognition_config.py`
- Saída: `experiments/<nome_experimento>/` (checkpoints, logs, figures, final_model)

## Estrutura importante

- `config/` — arquivos de configuração e hiperparâmetros
- `src/` — código-fonte (backbones, data_loader, pipelines, models, utils)
- `experiments/` — pastas de experimentos com subpastas:
  - `checkpoints/` — modelos salvos por época (ex.: `epoch_01.keras`)
  - `logs/` — CSV/JSON com histórico de treino/validação
  - `figures/` — gráficos de loss/accuracy e outros plots
  - `final_model*.keras` — modelo final salvo

## Requisitos

Instale as dependências listadas em `requirements.txt` (está no diretório raiz deste subprojeto):

```bash
pip install -r requirements.txt
```

Principais bibliotecas: TensorFlow/Keras, OpenCV, scikit-image e ferramentas opcionais (RetinaFace) usadas somente na validação/filtragem.

## Quickstart — comandos principais

- Treinar (exemplo mínimo):

```bash
python train.py --dataset_path /caminho/para/dataset
```

- Validar um checkpoint / modelo:

```bash
python run_validation.py --model_path experiments/<exp>/checkpoints/epoch_10.keras --dataset_path /caminho/para/val
```

- Finetuning (exemplo):

```bash
python run_finetuning.py \
  --strategy 2 \
  --pretrained_model experiments/Resnet50_vgg_cropado_CelebA/checkpoints/epoch_30.keras \
  --dataset_path /dados/datasets/aligned_112x112/ \
  --output_dir experiments/finetuning_strategy2 \
  --epochs 30 \
  --batch_size 64
```

Consulte `FINETUNING.md` para opções detalhadas (estratégias, `--num_layers`, `--use_retinaface`, etc.).

## Onde ficam os artefatos e como são nomeados

- Checkpoints: `experiments/<nome>/checkpoints/epoch_XX_val_acc_YY.keras` — salvos a cada época quando configurado.
- Modelo final: `experiments/<nome>/final_model*.keras` ou `final_model_<strategy>.keras`.
- Logs: `experiments/<nome>/logs/*.csv` (histórico de loss/accuracy e lr).
- Figuras: `experiments/<nome>/figures/*` (gráficos PNG com histórico).
- Relatórios de avaliação: `evaluation_final_report/` (quando gerado pelos scripts de avaliação).

Observação: a estrutura exata pode variar conforme o `output_dir` passado aos scripts; adapte os caminhos conforme seu experimento.

## Como carregar um modelo salvo (exemplo)

```python
import tensorflow as tf
from src.models.heads import heads  # ajustar conforme implementação real

# Exemplo genérico: carregar modelo Keras com custom objects, se necessário
model = tf.keras.models.load_model('experiments/finetuning_strategy1/final_model_full_fine-tuning.keras',
                                   compile=False)

# Use model.predict(...) ou extraia embeddings conforme o pipeline de inferência.
```

## Plots e visualizações

- Os gráficos gerados (loss/accuracy) ficam em `experiments/<nome>/figures/`.
- Os logs CSV podem ser carregados com Pandas para gerar análises customizadas.

## Dicas rápidas

- Para testar rapidamente, use um dataset pequeno e `--epochs 5`.
- Se ocorrer OOM, reduza `--batch_size` ou use estratégia 2 (treinar menos camadas).
- Use `--use_retinaface` nas validações quando seu dataset tiver imagens sem rosto ou ruído (veja `FINETUNING.md`).

---

Se quiser, posso também gerar um README reduzido em inglês ou adicionar exemplos de comandos com paths absolutos do seu ambiente. Basta pedir.

```
