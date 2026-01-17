
# CosFace Face Recognition Framework

Framework para treinamento e validação de reconhecimento facial usando CosFace.



## Principais Recursos

- Treinamento e validação com CosFace (Margin Cosine Product)
- Métricas completas: ROC, Confusion Matrix, Accuracy, Precision, Recall, F1, AUC, EER, FAR, FRR
- Detecção de faces opcional via RetinaFace
- Visualizações automáticas dos resultados
- Early Stopping configurável
- Suporte a Multi-GPU


## Instalação

```bash
pip install -r requirements.txt
```



## Como Usar

### Treinamento

```bash
python train.py \
    --root data/train/vggface2_aligned \
    --database VggFace2 \
    --network resnet50 \
    --classifier CosFace \
    --val-dataset lfw \
    --val-root data/lfw/val \
    --epochs 30 \
    --batch-size 64
```

#### Com validação de faces (RetinaFace)
```bash
python train.py \
    --root data/train/vggface2_aligned \
    --database VggFace2 \
    --network resnet50 \
    --classifier CosFace \
    --val-dataset lfw \
    --val-root data/lfw/val \
    --use-retinaface-validation \
    --no-face-policy exclude \
    --epochs 30
```

### Validação

Edite os caminhos do modelo, pesos e dados de validação em `evaluate.py` ou `evaluate_original.py`.

```bash
python evaluate.py
# ou
python evaluate_original.py
```
Os resultados e gráficos serão salvos no diretório configurado.



## Arquitetura

- **Backbone:** ResNet50
- **Loss:** CosFace (MCP)
- **Embeddings:** 512 dimensões


## Datasets

**Treinamento:**
- WebFace, VggFace2, MS1M, VggFaceHQ

**Validação:**
- LFW, CelebA

**Estrutura esperada:**
```
data/
├── train/<dataset_name>/identity_x/img.jpg
└── lfw/val/<person_name>/<person_name>_0001.jpg
```


## Argumentos Principais

| Argumento                | Tipo   | Default                       | Descrição                                 |
|--------------------------|--------|-------------------------------|-------------------------------------------|
| `--root`                 | str    | `data/train/webface_112x112/` | Diretório de imagens de treino            |
| `--database`             | str    | `WebFace`                     | Dataset: WebFace, VggFace2, MS1M, VggFaceHQ |
| `--network`              | str    | `resnet50`                    | Arquitetura: resnet50                     |
| `--classifier`           | str    | `CosFace`                     | Loss function: CosFace                    |
| `--batch-size`           | int    | 512                           | Tamanho do batch                          |
| `--epochs`               | int    | 30                            | Número de épocas                          |
| `--lr`                   | float  | 0.1                           | Learning rate inicial                     |
| `--momentum`             | float  | 0.9                           | Momentum do SGD                           |
| `--weight-decay`         | float  | 5e-4                          | Weight decay                              |
| `--num-workers`          | int    | 8                             | Workers do DataLoader                     |
| `--lr-scheduler`         | str    | `MultiStepLR`                 | Tipo: MultiStepLR, StepLR                 |
| `--milestones`           | int[]  | `[10, 20, 25]`                | Épocas para reduzir LR (MultiStepLR)      |
| `--step-size`            | int    | 10                            | Período de decay (StepLR)                 |
| `--gamma`                | float  | 0.1                           | Fator multiplicativo de decay             |
| `--val-dataset`          | str    | `lfw`                         | Dataset de validação: lfw, celeba         |
| `--val-root`             | str    | `data/lfw/val`                | Diretório do dataset de validação         |
| `--val-threshold`        | float  | 0.35                          | Threshold de similaridade                 |
| `--save-path`            | str    | `weights`                      | Diretório para salvar checkpoints         |
| `--checkpoint`           | str    | None                          | Checkpoint para continuar treino          |
| `--world-size`           | int    | 1                             | Número de processos distribuídos          |
| `--local_rank`           | int    | 0                             | Rank local para treinamento distribuído   |


## Validação com RetinaFace (Opcional)

Adicione os argumentos abaixo para ativar a validação de faces:

| Argumento                        | Tipo   | Default                  | Descrição                                 |
|-----------------------------------|--------|--------------------------|-------------------------------------------|
| `--use-retinaface-validation`     | flag   | False                    | Habilita validação com RetinaFace         |
| `--no-face-policy`                | str    | `exclude`                | Política para imagens sem face            |
| `--retinaface-conf-threshold`     | float  | 0.5                      | Threshold de confiança do detector        |
| `--face-validation-cache-dir`     | str    | `face_validation_cache`  | Diretório de cache                        |

Funcionamento:
- Primeira época: valida todas as imagens e salva cache
- Épocas seguintes: usa cache para acelerar
- Relatório final: estatísticas detalhadas em JSON


## Outputs e Métricas

**Diretórios:**
```
weights/
├── resnet50_CosFace_best.ckpt
├── resnet50_CosFace_last.ckpt
├── metrics/
│   ├── epoch_001/lfw_roc_curve.png
│   ├── epoch_001/lfw_confusion_matrix.png
│   └── final_evaluation/face_validation_report.json
└── final_report/
    ├── training_curves.png
    ├── confusion_matrix_evolution.png
    ├── learning_rate_schedule.png
    ├── all_metrics_overview.png
    ├── face_validation_stats.png
    ├── training_history.json
    └── training_summary.txt
```

**Métricas:**
- Treinamento: Loss, Accuracy
- Validação: Accuracy, Precision, Recall, F1, AUC, EER, FAR, FRR, ROC, Confusion Matrix
- Face Validation: estatísticas detalhadas (se habilitado)


## Avaliação

Execute:
```bash
python evaluate.py
# ou
python evaluate_original.py
```
Para análises avançadas, utilize o notebook `1.Notebooks/Eval.ipynb`.



## Retomar Treinamento

```bash
python train.py \
    --checkpoint weights/resnet50_CosFace_last.ckpt \
    --root data/train/vggface2_aligned \
    --database VggFace2 \
    --network resnet50 \
    --classifier CosFace
```
O histórico de métricas é preservado automaticamente.



## Multi-GPU

```bash
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    train.py \
    --world-size 2 \
    --root data/train/vggface2_aligned \
    --database VggFace2 \
    --network resnet50 \
    --classifier CosFace
```


## Preprocessamento

Imagens devem ser 112x112 pixels, RGB, normalizadas com mean=(0.5, 0.5, 0.5) e std=(0.5, 0.5, 0.5). O framework faz resize e normalização automática.


## Licença

Projeto para fins de pesquisa.