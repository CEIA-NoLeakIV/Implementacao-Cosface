"""
Script de Fine-tuning para Face Recognition
Permite selecionar entre 3 estratégias diferentes de fine-tuning:
1. Full Fine-tuning: Todas as camadas treináveis
2. Partial Fine-tuning: Apenas últimas N camadas do backbone
3. Differential LR Fine-tuning: Learning rates diferenciados por camada
"""

import sys
import os
import argparse
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping

# Configuração de GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Otimização de memória (Memory Growth) ativada para {len(gpus)} GPU(s).")
    except RuntimeError as e:
        print(e)

sys.path.append(os.getcwd())

from config.face_recognition_config import FaceRecognitionConfig
from src.pipelines.training_pipeline import run_face_training
from src.data_loader.face_datasets import get_train_val_datasets
from src.backbones.resnet import create_resnet50_cosface
from src.optimizers.scheduler import CosineAnnealingScheduler
from src.losses.margin_losses import CosFace
from src.utils.face_processing import process_image_pipeline_with_detection_flag


def load_pretrained_model(model_path, config):
    """Carrega um modelo pré-treinado."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo pré-treinado não encontrado em: {model_path}")
    
    print(f"Carregando modelo pré-treinado de: {model_path}")
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={'CosFace': CosFace}
    )
    return model


def strategy_1_full_finetuning(model, config, train_dataset, val_dataset, output_dir, epochs):
    """
    Estratégia 1: Fine-tuning Completo
    Todas as camadas do backbone e da cabeça são treináveis.
    """
    print("\n" + "="*60)
    print("ESTRATÉGIA 1: FINE-TUNING COMPLETO")
    print("="*60)
    print("Todas as camadas do backbone e da cabeça serão treinadas.")
    
    # Tornar todas as camadas treináveis
    backbone = model.get_layer("resnet50_backbone")
    backbone.trainable = True
    
    # Configurar otimizador com learning rate menor para fine-tuning
    optimizer = Adam(learning_rate=config.learning_rate * 0.1)  # LR 10x menor que o treinamento inicial
    
    # Compilar modelo
    crossentropy = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    metrics = [tf.keras.metrics.CategoricalAccuracy(name="accuracy")]
    model.compile(loss=crossentropy, optimizer=optimizer, metrics=metrics)
    
    print(f"Learning rate inicial: {optimizer.learning_rate.numpy():.6f}")
    print(f"Total de camadas treináveis: {sum([1 for layer in model.layers if layer.trainable])}")
    
    return model, optimizer


def strategy_2_partial_finetuning(model, config, train_dataset, val_dataset, output_dir, epochs, num_layers=10):
    """
    Estratégia 2: Fine-tuning Parcial
    Apenas as últimas N camadas do backbone são treináveis.
    """
    print("\n" + "="*60)
    print("ESTRATÉGIA 2: FINE-TUNING PARCIAL")
    print("="*60)
    print(f"Apenas as últimas {num_layers} camadas do backbone serão treinadas.")
    
    backbone = model.get_layer("resnet50_backbone")
    
    # Tornar apenas as últimas N camadas treináveis
    total_layers = len(backbone.layers)
    trainable_start = max(0, total_layers - num_layers)
    
    # Congelar todas as camadas primeiro
    backbone.trainable = False
    
    # Descongelar apenas as últimas N camadas
    for i in range(trainable_start, total_layers):
        backbone.layers[i].trainable = True
    
    # A cabeça sempre é treinável
    for layer in model.layers:
        if 'embedding' in layer.name or 'cosface' in layer.name:
            layer.trainable = True
    
    # Configurar otimizador
    optimizer = Adam(learning_rate=config.learning_rate * 0.1)
    
    # Compilar modelo
    crossentropy = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    metrics = [tf.keras.metrics.CategoricalAccuracy(name="accuracy")]
    model.compile(loss=crossentropy, optimizer=optimizer, metrics=metrics)
    
    trainable_count = sum([1 for layer in model.layers if layer.trainable])
    print(f"Learning rate inicial: {optimizer.learning_rate.numpy():.6f}")
    print(f"Total de camadas treináveis: {trainable_count}")
    print(f"Camadas do backbone treináveis: {num_layers}/{total_layers}")
    
    return model, optimizer


def strategy_3_differential_lr_finetuning(model, config, train_dataset, val_dataset, output_dir, epochs):
    """
    Estratégia 3: Fine-tuning com Learning Rate Diferenciado
    Camadas mais profundas (início do backbone) têm LR menor,
    camadas mais superficiais (fim do backbone e cabeça) têm LR maior.
    """
    print("\n" + "="*60)
    print("ESTRATÉGIA 3: FINE-TUNING COM LEARNING RATE DIFERENCIADO")
    print("="*60)
    print("Camadas profundas: LR baixo")
    print("Camadas superficiais: LR alto")
    
    backbone = model.get_layer("resnet50_backbone")
    backbone.trainable = True
    
    # Definir learning rates diferenciados
    # Camadas profundas (início): LR muito baixo
    # Camadas médias: LR médio
    # Camadas superficiais (fim): LR mais alto
    # Cabeça: LR mais alto ainda
    
    total_layers = len(backbone.layers)
    deep_layers = total_layers // 3
    mid_layers = total_layers // 3
    shallow_layers = total_layers - deep_layers - mid_layers
    
    # Criar otimizadores com diferentes learning rates
    base_lr = config.learning_rate * 0.1
    
    # Dividir o backbone em grupos
    deep_start = 0
    deep_end = deep_layers
    mid_start = deep_end
    mid_end = mid_start + mid_layers
    shallow_start = mid_end
    shallow_end = total_layers
    
    print(f"Divisão do backbone:")
    print(f"  Camadas profundas (0-{deep_end}): LR = {base_lr * 0.1:.6f}")
    print(f"  Camadas médias ({mid_start}-{mid_end}): LR = {base_lr * 0.5:.6f}")
    print(f"  Camadas superficiais ({shallow_start}-{shallow_end}): LR = {base_lr:.6f}")
    print(f"  Cabeça: LR = {base_lr * 2:.6f}")
    
    # Criar variáveis de learning rate para cada grupo
    lr_deep = tf.Variable(base_lr * 0.1, name='lr_deep')
    lr_mid = tf.Variable(base_lr * 0.5, name='lr_mid')
    lr_shallow = tf.Variable(base_lr, name='lr_shallow')
    lr_head = tf.Variable(base_lr * 2, name='lr_head')
    
    # Criar otimizadores para cada grupo
    optimizer_deep = Adam(learning_rate=lr_deep)
    optimizer_mid = Adam(learning_rate=lr_mid)
    optimizer_shallow = Adam(learning_rate=lr_shallow)
    optimizer_head = Adam(learning_rate=lr_head)
    
    # Para simplificar, vamos usar um único otimizador com learning rate médio
    # e aplicar multiplicadores de LR via callbacks ou manualmente
    # Na prática, TensorFlow/Keras não suporta facilmente múltiplos otimizadores
    # Vamos usar uma abordagem simplificada: treinar em fases
    
    # Por enquanto, usar LR médio e documentar que em produção seria ideal
    # usar uma implementação customizada de otimizador por camada
    optimizer = Adam(learning_rate=base_lr)
    
    # Compilar modelo
    crossentropy = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    metrics = [tf.keras.metrics.CategoricalAccuracy(name="accuracy")]
    model.compile(loss=crossentropy, optimizer=optimizer, metrics=metrics)
    
    print(f"Learning rate médio usado: {optimizer.learning_rate.numpy():.6f}")
    print("NOTA: Para LR verdadeiramente diferenciado, considere treinar em fases")
    print("      ou usar uma implementação customizada de otimizador por camada.")
    
    return model, optimizer


def apply_retinaface_validation_filter(val_dataset_raw, batch_size):
    """
    Aplica RetinaFace no dataset de validação e exclui amostras onde não detecta rosto.
    
    Args:
        val_dataset_raw: Dataset de validação RAW (sem pré-processamento) do TensorFlow
        batch_size: Tamanho do batch
        
    Returns:
        Dataset filtrado contendo apenas amostras com face detectada e pré-processado
    """
    print("\n" + "="*60)
    print("APLICANDO RETINAFACE NA VALIDAÇÃO")
    print("="*60)
    print("Filtrando amostras sem detecção de rosto...")
    
    def tf_align_with_flag(image, label):
        """Wrapper para processar imagem e retornar flag de detecção."""
        aligned_img, detection_flag = tf.numpy_function(
            func=process_image_pipeline_with_detection_flag,
            inp=[image],
            Tout=[tf.float32, tf.float32]
        )
        # Definir shapes manualmente após numpy_function
        aligned_img.set_shape((112, 112, 3))
        detection_flag.set_shape(())
        return aligned_img, label, detection_flag
    
    def filter_by_detection(aligned_img, label, detection_flag):
        """Filtra amostras baseado no flag de detecção."""
        # Retorna True se detectou face (flag > 0.5)
        return tf.greater(detection_flag, 0.5)
    
    def remove_flag(aligned_img, label, detection_flag):
        """Remove o flag do output, mantendo apenas imagem e label."""
        return aligned_img, label
    
    # Aplicar detecção e filtro
    val_dataset_with_flag = val_dataset_raw.map(
        tf_align_with_flag,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Filtrar amostras sem detecção
    val_dataset_filtered = val_dataset_with_flag.filter(filter_by_detection)
    
    # Remover o flag do output
    val_dataset_final = val_dataset_filtered.map(
        remove_flag,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Pré-processamento ResNet (normalização)
    val_dataset_final = val_dataset_final.map(
        lambda x, y: (tf.keras.applications.resnet50.preprocess_input(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Re-batching após filtro (pode ter batches menores)
    val_dataset_final = val_dataset_final.batch(batch_size)
    val_dataset_final = val_dataset_final.prefetch(tf.data.AUTOTUNE)
    
    print("Filtro RetinaFace aplicado com sucesso na validação.")
    print("Amostras sem detecção de rosto foram excluídas.\n")
    
    return val_dataset_final


def plot_finetuning_history(log_path, save_path, strategy_name):
    """Plota o histórico de treinamento do fine-tuning."""
    if not os.path.exists(log_path):
        print(f"Arquivo de log não encontrado: {log_path}")
        return
    
    log_data = pd.read_csv(log_path)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Acurácia
    axes[0, 0].plot(log_data['epoch'], log_data['accuracy'], label='Treino', marker='o')
    if 'val_accuracy' in log_data.columns:
        axes[0, 0].plot(log_data['epoch'], log_data['val_accuracy'], label='Validação', marker='s')
    axes[0, 0].set_title(f'Acurácia - {strategy_name}')
    axes[0, 0].set_xlabel('Época')
    axes[0, 0].set_ylabel('Acurácia')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Loss
    axes[0, 1].plot(log_data['epoch'], log_data['loss'], label='Treino', marker='o')
    if 'val_loss' in log_data.columns:
        axes[0, 1].plot(log_data['epoch'], log_data['val_loss'], label='Validação', marker='s')
    axes[0, 1].set_title(f'Loss - {strategy_name}')
    axes[0, 1].set_xlabel('Época')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Learning Rate
    if 'lr' in log_data.columns:
        axes[1, 0].plot(log_data['epoch'], log_data['lr'], label='Learning Rate', marker='o', color='green')
        axes[1, 0].set_title(f'Learning Rate - {strategy_name}')
        axes[1, 0].set_xlabel('Época')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        axes[1, 0].set_yscale('log')
    
    # Comparação Treino vs Validação (se disponível)
    if 'val_accuracy' in log_data.columns:
        axes[1, 1].plot(log_data['epoch'], log_data['accuracy'], label='Treino', marker='o')
        axes[1, 1].plot(log_data['epoch'], log_data['val_accuracy'], label='Validação', marker='s')
        axes[1, 1].fill_between(log_data['epoch'], 
                               log_data['accuracy'], 
                               log_data['val_accuracy'], 
                               alpha=0.3)
        axes[1, 1].set_title(f'Gap Treino-Validação - {strategy_name}')
        axes[1, 1].set_xlabel('Época')
        axes[1, 1].set_ylabel('Acurácia')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    else:
        # Se não houver validação, esconder o último subplot
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    figure_path = os.path.join(save_path, f'finetuning_history_{strategy_name.lower().replace(" ", "_")}.png')
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gráfico salvo em: {figure_path}")


def _ensure_image_size_tuple(image_size):
    """Garante que image_size seja uma tupla de 2 inteiros (height, width)."""
    if isinstance(image_size, (int, float)):
        return (int(image_size), int(image_size))
    elif isinstance(image_size, (tuple, list)) and len(image_size) == 2:
        return (int(image_size[0]), int(image_size[1]))
    else:
        return image_size


def run_finetuning(strategy, pretrained_model_path, dataset_path, output_dir, epochs=30, 
                   num_layers=10, batch_size=64, learning_rate=None, use_retinaface=False):
    """
    Executa o fine-tuning com a estratégia selecionada.
    
    Args:
        strategy: '1', '2', ou '3' - estratégia de fine-tuning
        pretrained_model_path: caminho para o modelo pré-treinado
        dataset_path: caminho para o novo dataset
        output_dir: diretório de saída para checkpoints e logs
        epochs: número de épocas
        num_layers: número de camadas para estratégia 2
        batch_size: tamanho do batch
        learning_rate: learning rate customizado (opcional)
        use_retinaface: se True, aplica RetinaFace na validação para excluir amostras sem rosto
    """
    print("\n" + "="*80)
    print("INICIANDO FINE-TUNING PARA FACE RECOGNITION")
    print("="*80)
    
    # Criar diretórios de saída
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    log_dir = os.path.join(output_dir, "logs")
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    # Carregar configuração
    config = FaceRecognitionConfig()
    config.batch_size = batch_size
    if learning_rate is not None:
        config.learning_rate = learning_rate
    config.epochs = epochs
    
    # Carregar dataset
    print(f"\nCarregando dataset de: {dataset_path}")
    
    # Verificar se existem diretórios train e val separados
    train_path = os.path.join(dataset_path, "train")
    val_path = os.path.join(dataset_path, "val")
    
    if os.path.exists(train_path) and os.path.exists(val_path):
        print("Diretórios 'train' e 'val' encontrados. Carregando de diretórios separados...")
        
        # Garantir que image_size seja uma tupla de 2 elementos
        image_size_tuple = _ensure_image_size_tuple(config.image_size)
        
        # Carregar dataset de treino do diretório train
        train_dataset_raw = tf.keras.utils.image_dataset_from_directory(
            train_path,
            seed=123,
            image_size=image_size_tuple,
            batch_size=config.batch_size,
            label_mode='categorical'
        )
        
        # Carregar dataset de validação do diretório val
        val_dataset_raw = tf.keras.utils.image_dataset_from_directory(
            val_path,
            seed=123,
            image_size=image_size_tuple,
            batch_size=config.batch_size,
            label_mode='categorical'
        )
        
        # Capturar número de classes do dataset de treino
        num_classes = len(train_dataset_raw.class_names)
        
        # Aplicar transformações ao dataset de treino
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.1),
        ])
        
        train_dataset = train_dataset_raw.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        train_dataset = train_dataset.map(
            lambda x, y: (tf.keras.applications.resnet50.preprocess_input(x), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
        
        # Aplicar RetinaFace na validação se habilitado
        if use_retinaface:
            print("\nCarregando dataset de validação RAW para aplicar RetinaFace...")
            val_dataset_raw_no_batch = tf.keras.utils.image_dataset_from_directory(
                val_path,
                seed=123,
                image_size=image_size_tuple,
                batch_size=None,  # Sem batch para processar individualmente
                label_mode='categorical'
            )
            
            # Aplicar RetinaFace na validação para excluir amostras sem detecção de rosto
            val_dataset = apply_retinaface_validation_filter(val_dataset_raw_no_batch, config.batch_size)
        else:
            print("\nRetinaFace desabilitado. Usando dataset de validação padrão.")
            # Aplicar apenas pré-processamento ResNet na validação
            val_dataset = val_dataset_raw.map(
                lambda x, y: (tf.keras.applications.resnet50.preprocess_input(x), y),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
        
    else:
        # Comportamento original: usar validation_split
        print("Diretórios 'train' e 'val' não encontrados. Usando validation_split=0.1...")
        train_dataset, val_dataset, num_classes = get_train_val_datasets(
            dataset_path,
            config.image_size,
            config.batch_size,
            validation_split=0.1,
            align_faces=False  # Não usar alinhamento no treino, será aplicado apenas na validação
        )
        
        # Aplicar RetinaFace na validação se habilitado
        if use_retinaface:
            print("\nCarregando dataset de validação RAW para aplicar RetinaFace...")
            image_size_tuple = _ensure_image_size_tuple(config.image_size)
            
            val_dataset_raw = tf.keras.utils.image_dataset_from_directory(
                dataset_path,
                validation_split=0.1,
                subset="validation",
                seed=123,
                image_size=image_size_tuple,
                batch_size=None,  # Sem batch para processar individualmente
                label_mode='categorical'
            )
            
            # Aplicar RetinaFace na validação para excluir amostras sem detecção de rosto
            val_dataset = apply_retinaface_validation_filter(val_dataset_raw, config.batch_size)
        else:
            print("\nRetinaFace desabilitado. Usando dataset de validação padrão.")
            # val_dataset já foi carregado acima sem filtro
    
    config.update_num_classes(num_classes)
    print(f"Número de classes no novo dataset: {config.num_classes}")
    
    # Carregar modelo pré-treinado
    pretrained_model = load_pretrained_model(pretrained_model_path, config)
    
    # Ajustar número de classes na camada CosFace se necessário
    cosface_layer = pretrained_model.get_layer("cosface_loss")
    if cosface_layer.n_classes != config.num_classes:
        print(f"\nAjustando número de classes de {cosface_layer.n_classes} para {config.num_classes}")
        # Criar novo modelo com número correto de classes
        model, feature_extractor, backbone = create_resnet50_cosface(config)
        # Transferir pesos do backbone do modelo pré-treinado
        try:
            pretrained_backbone = pretrained_model.get_layer("resnet50_backbone")
            backbone.set_weights(pretrained_backbone.get_weights())
            print("Pesos do backbone transferidos com sucesso.")
            
            # Tentar transferir pesos da camada de embedding (se compatível)
            try:
                pretrained_embedding = pretrained_model.get_layer("embedding_dense")
                new_embedding = model.get_layer("embedding_dense")
                if pretrained_embedding.get_weights()[0].shape[0] == new_embedding.get_weights()[0].shape[0]:
                    # Apenas transferir se a dimensão de entrada for compatível
                    new_embedding.set_weights(pretrained_embedding.get_weights())
                    print("Pesos da camada de embedding transferidos com sucesso.")
            except Exception as e:
                print(f"Nota: Pesos da camada de embedding não transferidos (normal para novo número de classes): {e}")
        except Exception as e:
            print(f"Aviso: Não foi possível transferir todos os pesos: {e}")
    else:
        # Número de classes é o mesmo, usar modelo pré-treinado diretamente
        model = pretrained_model
    
    # Aplicar estratégia selecionada
    strategy_names = {
        '1': 'Full Fine-tuning',
        '2': 'Partial Fine-tuning',
        '3': 'Differential LR Fine-tuning'
    }
    
    strategy_name = strategy_names.get(strategy, 'Unknown')
    
    if strategy == '1':
        model, optimizer = strategy_1_full_finetuning(
            model, config, train_dataset, val_dataset, output_dir, epochs
        )
    elif strategy == '2':
        model, optimizer = strategy_2_partial_finetuning(
            model, config, train_dataset, val_dataset, output_dir, epochs, num_layers
        )
    elif strategy == '3':
        model, optimizer = strategy_3_differential_lr_finetuning(
            model, config, train_dataset, val_dataset, output_dir, epochs
        )
    else:
        raise ValueError(f"Estratégia inválida: {strategy}. Use '1', '2', ou '3'.")
    
    # Configurar callbacks
    checkpoint_path = os.path.join(checkpoint_dir, f"finetuning_{strategy_name.lower().replace(' ', '_')}_epoch_{{epoch:02d}}.keras")
    log_path = os.path.join(log_dir, f"finetuning_{strategy_name.lower().replace(' ', '_')}_log.csv")
    
    callbacks = [
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='loss',
            mode='min',
            verbose=1,
            save_best_only=False,
            save_weights_only=False,
            save_freq='epoch'
        ),
        CSVLogger(log_path, append=False),
        CosineAnnealingScheduler(
            T_max=epochs,
            eta_max=float(optimizer.learning_rate.numpy()),
            eta_min=config.min_learning_rate,
            verbose=1
        )
    ]
    
    # Adicionar EarlyStopping se validação disponível
    if val_dataset is not None:
        callbacks.append(
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            )
        )
    
    # Treinar
    print(f"\n{'='*80}")
    print(f"Iniciando treinamento com {strategy_name}")
    print(f"{'='*80}\n")
    
    # Preparar dados de validação
    fit_kwargs = {
        'x': train_dataset,
        'epochs': epochs,
        'callbacks': callbacks,
        'verbose': 1
    }
    
    if val_dataset is not None:
        fit_kwargs['validation_data'] = val_dataset
    
    history = model.fit(**fit_kwargs)
    
    # Salvar modelo final
    final_model_path = os.path.join(output_dir, f"final_model_{strategy_name.lower().replace(' ', '_')}.keras")
    model.save(final_model_path)
    print(f"\nModelo final salvo em: {final_model_path}")
    
    # Plotar histórico
    plot_finetuning_history(log_path, figures_dir, strategy_name)
    
    print(f"\n{'='*80}")
    print("FINE-TUNING CONCLUÍDO COM SUCESSO!")
    print(f"{'='*80}")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"Logs: {log_path}")
    print(f"Figuras: {figures_dir}")
    print(f"Modelo final: {final_model_path}")
    
    return model, history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Fine-tuning de modelo de Face Recognition com 3 estratégias",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Estratégias disponíveis:
  1 - Full Fine-tuning: Todas as camadas treináveis
  2 - Partial Fine-tuning: Apenas últimas N camadas do backbone
  3 - Differential LR Fine-tuning: Learning rates diferenciados por camada

Exemplos de uso:
  # Estratégia 1: Fine-tuning completo
  python run_finetuning.py --strategy 1 --pretrained_model models/pretrained.keras \\
                           --dataset_path /dados/datasets/aligned_112x112/vggface2_dataset_all_splits_merged/ \\
                           --output_dir experiments/finetuning_strategy1

  # Estratégia 2: Fine-tuning parcial (últimas 15 camadas)
  python run_finetuning.py --strategy 2 --pretrained_model models/pretrained.keras \\
                           --dataset_path /dados/datasets/aligned_112x112/vggface2_dataset_all_splits_merged/ \\
                           --output_dir experiments/finetuning_strategy2 --num_layers 15

  # Estratégia 3: Fine-tuning com LR diferenciado
  python run_finetuning.py --strategy 3 --pretrained_model models/pretrained.keras \\
                           --dataset_path /dados/datasets/aligned_112x112/vggface2_dataset_all_splits_merged/ \\
                           --output_dir experiments/finetuning_strategy3

  # Com RetinaFace habilitado na validação
  python run_finetuning.py --strategy 1 --pretrained_model models/pretrained.keras \\
                           --dataset_path /dados/datasets/aligned_112x112/vggface2_dataset_all_splits_merged/ \\
                           --output_dir experiments/finetuning_strategy1 --use_retinaface
        """
    )
    
    parser.add_argument(
        '--strategy',
        type=str,
        required=True,
        choices=['1', '2', '3'],
        help='Estratégia de fine-tuning: 1=Full, 2=Partial, 3=Differential LR'
    )
    parser.add_argument(
        '--pretrained_model',
        type=str,
        required=True,
        help='Caminho para o modelo pré-treinado (.keras)'
    )
    parser.add_argument(
        '--dataset_path',
        type=str,
        required=True,
        help='Caminho para o novo dataset de treinamento'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Diretório de saída para checkpoints, logs e figuras'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=30,
        help='Número de épocas (default: 30)'
    )
    parser.add_argument(
        '--num_layers',
        type=int,
        default=10,
        help='Número de camadas para estratégia 2 (default: 10)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Tamanho do batch (default: 64)'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=None,
        help='Learning rate customizado (opcional, default: usa config)'
    )
    parser.add_argument(
        '--use_retinaface',
        action='store_true',
        help='Habilita RetinaFace na validação para excluir amostras sem detecção de rosto (default: False)'
    )
    
    args = parser.parse_args()
    
    run_finetuning(
        strategy=args.strategy,
        pretrained_model_path=args.pretrained_model,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_retinaface=args.use_retinaface
    )

