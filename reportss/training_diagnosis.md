# Diagnóstico do Treinamento

## Sintomas observados

* **Loss extremamente alta desde o início** e **acurácia próxima de zero**.
* Instabilidade logo nas primeiras épocas, resultando em `NaN` com facilidade.
* Callbacks de validação LFW não refletiam o desempenho real do modelo porque o
  pré-processamento durante a validação não era consistente com o utilizado no
  treino.

## Causas Raiz

1. **Código em estado inconsistente**: arquivos fundamentais (`config`,
   `training_pipeline`, `backbones` e `data_loader`) continham marcações de
   _merge_ (`<<<<<<<`, `=======`, `>>>>>>>`). Isso deixava o repositório em um
   estado indefinido, misturando duas estratégias distintas de treinamento e
   hiperparâmetros conflitantes (paths absolutos, fração do dataset, agendamento
   de _learning rate_ etc.). Na prática, a pipeline carregava configurações
   incompatíveis e o comportamento mudava a cada execução.
2. **Treinamento agressivo da CosFace**: o modelo começava ajustando tanto o
   `Dense` de embeddings quanto camadas do backbone com CosFace + SGD logo nas
   primeiras iterações. Sem um _warm-up_ isso gera gradientes muito grandes nas
   últimas camadas da ResNet50, fazendo a loss explodir e impedindo que a
   acurácia saia de zero.
3. **Validação inconsistente**: durante o callback de LFW as imagens eram
   normalizadas em `[0, 1]`, diferente do `preprocess_input` usado no treino.
   Isso distorcia as métricas de verificação, dificultando a detecção de
   melhorias reais.

## Correções Aplicadas

* Remoção completa das marcações de _merge_ e consolidação de uma única versão
  da pipeline (configuração baseada em dataclass, paths flexíveis via variável de
  ambiente e documentação dos hiperparâmetros).
* Nova rotina de treinamento em duas fases:
  1. _Warm-up_ com o backbone congelado e otimizador Adam para ajustar apenas a
     cabeça CosFace com `label_smoothing`.
  2. _Fine-tuning_ liberando apenas os últimos blocos da ResNet50, otimizando
     com SGD + `CosineAnnealingScheduler`, _early stopping_ em `val_lfw_accuracy`
     e `TerminateOnNaN` para interromper divergências.
* Cabeça de embeddings mais estável (BatchNorm + Dropout) e parâmetros do CosFace
  expostos na configuração.
* Pipeline de dados consistente: mesma normalização (`preprocess_input`) para
  treino, validação e callback de LFW, além de busca do melhor _threshold_ para a
  métrica.

## Próximos Passos Recomendados

* Ajustar `DATA_ROOT` ou exportar a variável de ambiente correspondente antes do
  treino para apontar para o local correto do VGGFace2/LFW.
* Monitorar `deliverables/face_recognition_log.csv` e os gráficos em
  `reports/figures` para verificar se a loss cai gradualmente após o warm-up.
* Caso a acurácia ainda fique baixa, reduza `dataset_fraction` para um valor
  menor (0.1 ou 0.2) e valide se a pipeline converge com um subconjunto antes de
  escalar para 100% das classes.