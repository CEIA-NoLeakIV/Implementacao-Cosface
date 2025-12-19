# Guia de Fine-tuning para Face Recognition

Este documento apresenta um guia completo para realizar fine-tuning do modelo de Face Recognition treinado no VGGFace2 em um novo conjunto de dados.

## 📋 Índice

1. [Visão Geral](#visão-geral)
2. [Pré-requisitos](#pré-requisitos)
3. [Estrutura do Dataset](#estrutura-do-dataset)
4. [RetinaFace na Validação](#retinaface-na-validação)
5. [Estratégias de Fine-tuning](#estratégias-de-fine-tuning)
6. [Passo a Passo](#passo-a-passo)
7. [Salvamento de Modelos](#salvamento-de-modelos)
8. [Monitoramento e Avaliação](#monitoramento-e-avaliação)
9. [Troubleshooting](#troubleshooting)

---

## 🎯 Visão Geral

O fine-tuning permite adaptar um modelo pré-treinado (treinado no VGGFace2 e validado no LFW) para um novo conjunto de dados específico. Este processo é mais eficiente que treinar do zero e geralmente produz melhores resultados com menos dados.

### Quando usar Fine-tuning?

- ✅ Você tem um novo conjunto de dados com classes diferentes do VGGFace2
- ✅ Você tem um conjunto de dados menor (< 100k imagens)
- ✅ Você quer adaptar o modelo para um domínio específico
- ✅ Você quer manter as características aprendidas do modelo original

---

## 📦 Pré-requisitos

### 1. Modelo Pré-treinado

Você precisa ter um modelo pré-treinado salvo. O modelo deve ser do tipo `.keras` e conter:
- Backbone ResNet50
- Camada CosFace
- Pesos treinados no VGGFace2

**Localização esperada:** O modelo pode estar em qualquer local. Exemplos:
- `experiments/Resnet50_vgg_cropado_CelebA/checkpoints/epoch_XX.keras`
- `models/pretrained_model.keras`

### 2. Novo Dataset

O novo dataset deve estar organizado em uma das seguintes estruturas:

#### Opção A: Dataset com divisão train/val
```
/dados/datasets/aligned_112x112/vggface2_dataset_all_splits_merged/
├── train/
│   ├── pessoa_001/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   ├── pessoa_002/
│   │   └── ...
│   └── ...
└── val/
    ├── pessoa_001/
    ├── pessoa_002/
    └── ...
```

#### Opção B: Dataset único (será dividido automaticamente)
```
/dados/datasets/aligned_112x112/vggface2_dataset_all_splits_merged/
├── pessoa_001/
│   ├── img1.jpg
```markdown
# Guia prático de Fine-tuning — Cosface

Este documento traz instruções concisas e prontas para executar fine-tuning de modelos pré-treinados, incluindo padrões de entrada, estratégias recomendadas, flags importantes e onde os artefatos são salvos.

## Rápido: pré-requisitos

- Modelo pré-treinado em formato `.keras` (ex.: `experiments/.../epoch_XX.keras`)
- Dataset organizado por pastas (cada pasta = 1 classe) ou com `train/` e `val/` separados
- Dependências: `pip install -r requirements.txt`

Recomendação: imagens alinhadas em 112x112 (como usado no pipeline). Mínimo sugerido: ~10 imagens/por classe.

## Comando essencial

Exemplo genérico:

```bash
python run_finetuning.py \
  --strategy 2 \
  --pretrained_model /caminho/para/pretrained.keras \
  --dataset_path /caminho/para/dataset \
  --output_dir experiments/finetuning_experiment \
  --epochs 30 \
  --batch_size 64
```

Flags úteis:
- `--strategy`: 1 (full), 2 (partial), 3 (differential LR)
- `--num_layers`: (para strategy 2) número de camadas finais a descongelar
- `--use_retinaface`: habilita filtragem por detecção de rosto na validação
- `--learning_rate`, `--batch_size`, `--epochs`, `--output_dir`

## Estratégias (resumo)

- Strategy 1 — Full: descongela todo o backbone; use com datasets grandes.
- Strategy 2 — Partial: descongela apenas últimas N camadas; indicado para datasets médios/pequenos.
- Strategy 3 — Differential LR: todas camadas treináveis com políticas de LR (usar com cuidado).

Recomendações iniciais: começar com `strategy 2` e `--num_layers 10`, `--epochs 20-30`, `--batch_size 32/64`.

## RetinaFace na validação

Opção: `--use_retinaface` — aplica detecção de faces apenas na fase de validação para excluir imagens sem rosto detectado.

Comportamento:
- Se habilitado, imagens sem detecção são excluídas da validação. Útil quando o dataset contém ruído ou imagens sem rosto.
- Se desabilitado, validação usa todas as amostras.

Exemplo habilitando RetinaFace:

```bash
python run_finetuning.py --strategy 1 --pretrained_model /caminho/model.keras --dataset_path /dados --output_dir experiments/fin1 --use_retinaface
```

## Saída esperada (por `--output_dir`)

Estrutura típica criada pelo script:

```
experiments/<nome_experiment>/
├── checkpoints/            # modelos por época (epoch_01.keras ...)
├── logs/                   # CSV/JSON com histórico de treino/val
├── figures/                # PNGs com graphs (loss/acc)
└── final_model*.keras      # modelo final salvo
```

Use os arquivos em `checkpoints/` para retomar treinos ou para análises históricas.

## Como retomar/continuar de um checkpoint

Se quiser retomar, passe o checkpoint como `--pretrained_model` e ajuste hiperparâmetros.

Exemplo:

```bash
python run_finetuning.py --strategy 2 --pretrained_model experiments/finetuning_experiment/checkpoints/epoch_05.keras --dataset_path /dados --output_dir experiments/finetuning_resume --epochs 20
```

## Carregando modelo com objetos customizados

Se o modelo usar perdas/objetos customizados (ex.: CosFace), carregue com `custom_objects` no Keras:

```python
import tensorflow as tf
from src.losses.margin_losses import CosFace

model = tf.keras.models.load_model('experiments/finetuning_experiment/final_model.keras', custom_objects={'CosFace': CosFace}, compile=False)
```

## Troubleshooting rápido

- OOM: reduzir `--batch_size`, usar strategy 2 ou reduzir resolução.
- Perda não diminui: verificar loading do modelo pré-treinado, aumentar LR ou revisar dados.
- Muitas amostras excluídas pelo RetinaFace: revisar qualidade/align das imagens.

## Monitoramento

- Ver logs em: `experiments/<nome>/logs/` (CSV) — abrir com Pandas/Excel para análise.
- Ver plots em: `experiments/<nome>/figures/`.

---

Se quiser, eu deixo esse guia mais detalhado com exemplos de `config/face_recognition_config.py` e um snippet para gerar relatórios automáticos (CSV → PDF) — diga qual formato prefere.

```
- 💡 **Recomendação**: Use `--use_retinaface` quando o dataset pode conter imagens sem rosto ou de baixa qualidade


