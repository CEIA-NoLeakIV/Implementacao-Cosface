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
   - O treinamento é iniciado com o script `run_training.py`.
   - Exemplo de comando:
     ```bash
     python run_training.py --dataset_path /path/to/dataset
     ```

### Validação
1. **Pipeline de Validação:**
   - A validação é realizada separadamente após o treinamento, utilizando o script `run_validation.py`.
   - Suporta validação em múltiplos datasets, incluindo LFW.

2. **Script de Execução:**
   - Exemplo de comando:
     ```bash
     python run_validation.py --model_path /path/to/checkpoint --dataset_path /path/to/validation/dataset
     ```

---

## Componentes Principais

### Configurações (`config/`)
- Define hiperparâmetros como taxa de aprendizado, número de épocas, e caminhos para datasets e checkpoints.

### Pipelines (`src/pipelines/`)
- **`training_pipeline.py`:** Gerencia o treinamento do modelo.
- **`inference_pipeline.py`:** Gerencia a inferência em novos dados.

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

### `run_training.py`
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
   python run_training.py --dataset_path /path/to/dataset
   ```

2. **Validar um Modelo:**
   ```bash
   python run_validation.py --model_path /path/to/checkpoint --dataset_path /path/to/validation/dataset
   ```

3. **Realizar Fine-tuning:**
   ```bash
   python run_finetuning.py \
       --strategy 2 \
       --pretrained_model experiments/Resnet50_vgg_cropado_CelebA/checkpoints/epoch_30.keras \
       --dataset_path /dados/datasets/aligned_112x112/vggface2_dataset_all_splits_merged/ \
       --output_dir experiments/finetuning_strategy2 \
       --epochs 30 \
       --num_layers 10
   ```
   Consulte `FINETUNING.md` para guia completo e exemplos detalhados.

4. **Estrutura de Resultados:**
   - Checkpoints, logs e figuras são salvos na pasta `experiments/`.

---

Este documento serve como guia para entender e usar o framework Cosface. Para dúvidas ou melhorias, contribua diretamente no repositório!
