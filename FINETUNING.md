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
│   ├── img2.jpg
│   └── ...
├── pessoa_002/
│   └── ...
└── ...
```

**Requisitos:**
- Imagens devem estar alinhadas e redimensionadas para 112x112 pixels
- Formato: JPG ou PNG
- Cada pasta representa uma classe (pessoa)
- Mínimo recomendado: 10 imagens por classe

### 3. Dependências

Certifique-se de que todas as dependências estão instaladas:

```bash
pip install -r requirements.txt
```

**Dependências importantes:**
- TensorFlow/Keras
- RetinaFace (via `uniface`) - usado automaticamente na validação
- OpenCV
- scikit-image

---

## 🔍 RetinaFace na Validação

### Visão Geral

O script oferece a opção de aplicar o **RetinaFace** na fase de validação para garantir que apenas amostras com detecção de rosto sejam utilizadas. Isso melhora a qualidade da validação e evita que imagens sem rosto afetem as métricas.

**⚠️ Importante:** O RetinaFace é **opcional** e deve ser habilitado explicitamente usando a flag `--use_retinaface`.

### Como Funciona

1. **Durante o Treinamento:**
   - O dataset de treino é processado normalmente, sem filtro de RetinaFace
   - Isso permite que o modelo aprenda com todos os dados disponíveis

2. **Durante a Validação (quando `--use_retinaface` está habilitado):**
   - Cada imagem do dataset de validação passa pelo RetinaFace
   - Se o RetinaFace **não detectar** um rosto, a amostra é **excluída** automaticamente
   - Apenas amostras com detecção de rosto são usadas para calcular métricas de validação

3. **Quando RetinaFace está desabilitado:**
   - O dataset de validação é usado normalmente, sem filtro
   - Todas as amostras são incluídas na validação

### Política de Exclusão

- ✅ **Amostras com rosto detectado**: Incluídas na validação
- ❌ **Amostras sem rosto detectado**: Excluídas automaticamente
- ⚠️ **Erros de detecção**: Tratados como "sem detecção" e excluídos

### Benefícios

- **Validação mais precisa**: Apenas imagens válidas são consideradas
- **Métricas mais confiáveis**: Evita ruído de imagens sem rosto
- **Processo automático**: Não requer intervenção manual
- **Apenas na validação**: Não afeta o treinamento

### Como Habilitar

Para habilitar o RetinaFace na validação, adicione a flag `--use_retinaface` ao comando:

```bash
python run_finetuning.py \
    --strategy 1 \
    --pretrained_model models/pretrained.keras \
    --dataset_path /dados/datasets/... \
    --output_dir experiments/finetuning_strategy1 \
    --use_retinaface  # <-- Adicione esta flag
```

**Sem a flag:** O RetinaFace não é aplicado e todas as amostras de validação são usadas.

### Mensagens Durante a Execução

Quando o RetinaFace está habilitado, você verá mensagens como:

```
Carregando dataset de validação RAW para aplicar RetinaFace...
============================================================
APLICANDO RETINAFACE NA VALIDAÇÃO
============================================================
Filtrando amostras sem detecção de rosto...
Filtro RetinaFace aplicado com sucesso na validação.
Amostras sem detecção de rosto foram excluídas.
```

Quando desabilitado:
```
RetinaFace desabilitado. Usando dataset de validação padrão.
```

### Notas Importantes

- ⚠️ O RetinaFace é **opcional** e deve ser habilitado com `--use_retinaface`
- ⚠️ O RetinaFace é aplicado **apenas na validação**, não no treinamento
- ⚠️ O processo de filtragem pode reduzir o tamanho do dataset de validação
- ⚠️ Se muitas amostras forem excluídas, considere revisar a qualidade do dataset
- ✅ O modelo RetinaFace é carregado uma única vez e reutilizado (eficiente)
- 💡 **Recomendação**: Use `--use_retinaface` quando o dataset pode conter imagens sem rosto ou de baixa qualidade

---

## 🔧 Estratégias de Fine-tuning

O script oferece 3 estratégias diferentes, cada uma adequada para diferentes cenários:

### Estratégia 1: Full Fine-tuning (Fine-tuning Completo)

**Quando usar:**
- Você tem um dataset grande (> 10k imagens)
- O novo dataset é similar ao VGGFace2
- Você quer máxima adaptação ao novo domínio

**Características:**
- Todas as camadas do backbone são treináveis
- Todas as camadas da cabeça são treináveis
- Learning rate: 10% do learning rate original
- Mais flexível, mas requer mais dados

**Comando:**
```bash
python run_finetuning.py \
    --strategy 1 \
    --pretrained_model experiments/Resnet50_vgg_cropado_CelebA/checkpoints/epoch_30.keras \
    --dataset_path /dados/datasets/aligned_112x112/vggface2_dataset_all_splits_merged/ \
    --output_dir experiments/finetuning_strategy1 \
    --epochs 30 \
    --batch_size 64 \
    --use_retinaface  # Opcional: habilita filtro RetinaFace na validação
```

---

### Estratégia 2: Partial Fine-tuning (Fine-tuning Parcial)

**Quando usar:**
- Você tem um dataset médio (1k - 10k imagens)
- Quer evitar overfitting
- Quer preservar mais características do modelo original

**Características:**
- Apenas as últimas N camadas do backbone são treináveis
- Todas as camadas da cabeça são treináveis
- Camadas iniciais do backbone permanecem congeladas
- Menos parâmetros treináveis = menos risco de overfitting

**Parâmetros:**
- `--num_layers`: Número de camadas finais a treinar (default: 10)
  - Valores recomendados: 5-20
  - Mais camadas = mais adaptação, mas mais risco de overfitting

**Comando:**
```bash
python run_finetuning.py \
    --strategy 2 \
    --pretrained_model experiments/Resnet50_vgg_cropado_CelebA/checkpoints/epoch_30.keras \
    --dataset_path /dados/datasets/aligned_112x112/vggface2_dataset_all_splits_merged/ \
    --output_dir experiments/finetuning_strategy2 \
    --epochs 30 \
    --batch_size 64 \
    --num_layers 15
```

---

### Estratégia 3: Differential LR Fine-tuning (LR Diferenciado)

**Quando usar:**
- Você quer um equilíbrio entre adaptação e preservação
- Você tem experiência com fine-tuning
- Você quer máxima performance

**Características:**
- Todas as camadas são treináveis
- Learning rates diferenciados por profundidade:
  - Camadas profundas: LR muito baixo (preserva features básicas)
  - Camadas médias: LR médio
  - Camadas superficiais: LR mais alto (adapta features específicas)
  - Cabeça: LR mais alto ainda

**Nota:** A implementação atual usa um LR médio. Para LR verdadeiramente diferenciado, considere treinar em fases ou usar uma implementação customizada.

**Comando:**
```bash
python run_finetuning.py \
    --strategy 3 \
    --pretrained_model experiments/Resnet50_vgg_cropado_CelebA/checkpoints/epoch_30.keras \
    --dataset_path /dados/datasets/aligned_112x112/vggface2_dataset_all_splits_merged/ \
    --output_dir experiments/finetuning_strategy3 \
    --epochs 30 \
    --batch_size 64
```

---

## 📝 Passo a Passo

### Passo 1: Preparar o Ambiente

```bash
# Navegar para o diretório do projeto
cd /Users/wgalvao/Noleak/Implementacao-Cosface

# Verificar se o modelo pré-treinado existe
ls -lh experiments/Resnet50_vgg_cropado_CelebA/checkpoints/

# Verificar se o dataset está acessível
ls -lh /dados/datasets/aligned_112x112/vggface2_dataset_all_splits_merged/
```

### Passo 2: Verificar o Dataset

```bash
# Contar número de classes
find /dados/datasets/aligned_112x112/vggface2_dataset_all_splits_merged/ -type d -mindepth 1 -maxdepth 1 | wc -l

# Verificar estrutura (se tem train/val)
ls /dados/datasets/aligned_112x112/vggface2_dataset_all_splits_merged/
```

### Passo 3: Escolher a Estratégia

Considere:
- **Tamanho do dataset**: Pequeno → Estratégia 2, Grande → Estratégia 1
- **Similaridade com VGGFace2**: Similar → Estratégia 1, Diferente → Estratégia 2
- **Recursos computacionais**: Limitados → Estratégia 2, Abundantes → Estratégia 1

### Passo 4: Executar o Fine-tuning

**Exemplo completo com Estratégia 1:**

```bash
python run_finetuning.py \
    --strategy 1 \
    --pretrained_model experiments/Resnet50_vgg_cropado_CelebA/checkpoints/epoch_30.keras \
    --dataset_path /dados/datasets/aligned_112x112/vggface2_dataset_all_splits_merged/ \
    --output_dir experiments/finetuning_strategy1 \
    --epochs 30 \
    --batch_size 64 \
    --learning_rate 0.0005 \
    --use_retinaface  # Opcional: habilita filtro RetinaFace na validação
```

**Exemplo com Estratégia 2 (recomendado para começar):**

```bash
python run_finetuning.py \
    --strategy 2 \
    --pretrained_model experiments/Resnet50_vgg_cropado_CelebA/checkpoints/epoch_30.keras \
    --dataset_path /dados/datasets/aligned_112x112/vggface2_dataset_all_splits_merged/ \
    --output_dir experiments/finetuning_strategy2 \
    --epochs 30 \
    --batch_size 64 \
    --num_layers 10 \
    --use_retinaface  # Opcional: habilita filtro RetinaFace na validação
```

### Passo 5: Monitorar o Progresso

Durante o treinamento, você verá:
- Progresso por época
- Loss e acurácia de treino
- Loss e acurácia de validação (com ou sem filtro RetinaFace, dependendo da flag)
- Learning rate atual
- Mensagens sobre aplicação do RetinaFace na validação (se `--use_retinaface` estiver habilitado)

### Passo 6: Verificar Resultados

Após o treinamento, verifique:

```bash
# Verificar checkpoints salvos
ls -lh experiments/finetuning_strategy1/checkpoints/

# Verificar logs
head experiments/finetuning_strategy1/logs/finetuning_full_fine-tuning_log.csv

# Verificar figuras geradas
ls -lh experiments/finetuning_strategy1/figures/
```

---

## 💾 Salvamento de Modelos

### Estrutura de Saída

O script cria a seguinte estrutura:

```
experiments/finetuning_strategy1/
├── checkpoints/
│   ├── finetuning_full_fine-tuning_epoch_01.keras
│   ├── finetuning_full_fine-tuning_epoch_02.keras
│   └── ...
├── logs/
│   └── finetuning_full_fine-tuning_log.csv
├── figures/
│   └── finetuning_history_full_fine-tuning.png
└── final_model_full_fine-tuning.keras
```

### Tipos de Modelos Salvos

1. **Checkpoints por Época** (`checkpoints/`)
   - Um modelo salvo a cada época
   - Útil para análise posterior
   - Permite retomar de qualquer época

2. **Modelo Final** (`final_model_*.keras`)
   - Modelo da última época
   - Pronto para uso em produção
   - Contém todos os pesos otimizados

### Carregar Modelo Fine-tuned

```python
import tensorflow as tf
from src.losses.margin_losses import CosFace

# Carregar modelo
model = tf.keras.models.load_model(
    'experiments/finetuning_strategy1/final_model_full_fine-tuning.keras',
    custom_objects={'CosFace': CosFace}
)

# Usar para inferência
# ... seu código de inferência ...
```

### Salvar Apenas os Pesos

Se você quiser salvar apenas os pesos (mais leve):

```python
# Após o fine-tuning
model.save_weights('experiments/finetuning_strategy1/final_weights.h5')

# Para carregar depois
model.load_weights('experiments/finetuning_strategy1/final_weights.h5')
```

---

## 📊 Monitoramento e Avaliação

### Durante o Treinamento

O script gera automaticamente:

1. **Log CSV** (`logs/finetuning_*_log.csv`)
   - Contém: epoch, loss, accuracy, val_loss, val_accuracy, lr
   - Pode ser aberto no Excel/Pandas para análise

2. **Gráficos** (`figures/finetuning_history_*.png`)
   - Acurácia de treino e validação
   - Loss de treino e validação
   - Learning rate ao longo do tempo
   - Gap entre treino e validação

### Análise dos Resultados

**Sinais de bom fine-tuning:**
- ✅ Loss diminuindo consistentemente
- ✅ Acurácia aumentando
- ✅ Gap pequeno entre treino e validação (< 5%)
- ✅ Sem overfitting (validação acompanha treino)

**Sinais de problemas:**
- ❌ Loss não diminui ou aumenta
- ❌ Overfitting (treino muito melhor que validação)
- ❌ Acurácia estagnada
- ❌ Loss com NaN
- ❌ Muitas amostras excluídas pelo RetinaFace (verifique qualidade do dataset)

### Validação no LFW (Opcional)

Após o fine-tuning, você pode validar no LFW:

```bash
python run_validation.py \
    --model_path experiments/finetuning_strategy1/final_model_full_fine-tuning.keras \
    --dataset_path /path/to/validation/dataset \
    --lfw_path /path/to/lfw \
    --lfw_pairs /path/to/lfw_pairs.txt
```

---

## 🔍 Troubleshooting

### Problema: "Modelo pré-treinado não encontrado"

**Solução:**
```bash
# Verificar se o caminho está correto
ls -lh experiments/Resnet50_vgg_cropado_CelebA/checkpoints/

# Usar caminho absoluto
python run_finetuning.py \
    --pretrained_model /caminho/absoluto/para/modelo.keras \
    ...
```

### Problema: "Dataset não encontrado"

**Solução:**
```bash
# Verificar permissões
ls -lh /dados/datasets/aligned_112x112/vggface2_dataset_all_splits_merged/

# Verificar estrutura
find /dados/datasets/aligned_112x112/vggface2_dataset_all_splits_merged/ -type f | head -10
```

### Problema: "Número de classes diferente"

**Solução:**
O script ajusta automaticamente o número de classes. Se houver erro:
- Verifique se o dataset tem pelo menos 2 classes
- Verifique se as pastas estão organizadas corretamente

### Problema: "Out of Memory (OOM)"

**Soluções:**
1. Reduzir batch size:
   ```bash
   --batch_size 32  # ou 16, ou 8
   ```

2. Usar Estratégia 2 (menos parâmetros):
   ```bash
   --strategy 2 --num_layers 5
   ```

3. Reduzir tamanho da imagem (modificar config se necessário)

### Problema: "Loss não diminui"

**Soluções:**
1. Aumentar learning rate:
   ```bash
   --learning_rate 0.001
   ```

2. Verificar se o dataset está correto
3. Verificar se o modelo pré-treinado está carregado corretamente
4. Tentar Estratégia 1 (mais flexível)

### Problema: "Overfitting"

**Soluções:**
1. Usar Estratégia 2:
   ```bash
   --strategy 2 --num_layers 5
   ```

2. Reduzir learning rate:
   ```bash
   --learning_rate 0.0001
   ```

3. Adicionar mais dados de treinamento
4. Usar data augmentation (já incluído no pipeline)

### Problema: "Muitas amostras excluídas pelo RetinaFace"

**Sintomas:**
- Mensagem indicando muitas exclusões durante a validação
- Dataset de validação muito pequeno após filtro

**Soluções:**
1. Verificar qualidade das imagens no dataset:
   ```bash
   # Verificar algumas imagens manualmente
   find /dados/datasets/.../val -name "*.jpg" | head -10 | xargs -I {} open {}
   ```

2. Verificar se as imagens estão alinhadas corretamente
3. Considerar pré-processar o dataset antes do fine-tuning
4. Verificar se o RetinaFace está funcionando corretamente:
   - Teste com algumas imagens manualmente
   - Verifique logs de erro do RetinaFace

**Nota:** Algumas exclusões são normais, especialmente se o dataset contém imagens de baixa qualidade ou sem rosto visível.

---

## 📈 Recomendações Finais

### Para Iniciantes

1. Comece com **Estratégia 2** (`--strategy 2 --num_layers 10`)
2. Use `--epochs 20` para testes iniciais
3. Monitore os gráficos gerados
4. Ajuste `--num_layers` baseado nos resultados

### Para Experientes

1. Experimente todas as 3 estratégias
2. Compare resultados usando os logs CSV
3. Ajuste learning rates baseado no dataset
4. Considere treinar em múltiplas fases

### Boas Práticas

- ✅ Sempre valide em um conjunto de teste separado
- ✅ Salve checkpoints frequentes
- ✅ Monitore overfitting
- ✅ Documente os parâmetros usados
- ✅ Compare diferentes estratégias no mesmo dataset
- ✅ Verifique a qualidade do dataset antes do fine-tuning
- ✅ Monitore quantas amostras são excluídas pelo RetinaFace na validação
- ✅ Use o filtro RetinaFace para garantir validação com imagens válidas

---

## 📚 Referências

- [CosFace Paper](https://arxiv.org/abs/1801.09414)
- [Transfer Learning Guide](https://www.tensorflow.org/guide/keras/transfer_learning)
- [Fine-tuning Best Practices](https://cs231n.github.io/transfer-learning/)

---

## 🤝 Suporte

Para problemas ou dúvidas:
1. Verifique os logs em `experiments/finetuning_*/logs/`
2. Verifique os gráficos em `experiments/finetuning_*/figures/`
3. Consulte a seção de Troubleshooting acima

---

**Última atualização:** 2024

---

## 🔄 Histórico de Mudanças

### Versão Atual

- ✅ **RetinaFace Opcional na Validação**: Filtro de RetinaFace pode ser habilitado com `--use_retinaface`
- ✅ **Política de Exclusão**: Amostras sem detecção de rosto são automaticamente excluídas da validação (quando habilitado)
- ✅ **Processamento Eficiente**: RetinaFace é carregado uma única vez e reutilizado
- ✅ **Apenas na Validação**: Filtro aplicado apenas na validação, não afeta o treinamento
- ✅ **Controle Flexível**: Usuário decide quando usar o filtro através da flag `--use_retinaface`

