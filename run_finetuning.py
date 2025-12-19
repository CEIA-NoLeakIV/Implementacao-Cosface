"""
Script de Fine-tuning para Face Recognition
CORRIGIDO: 
1. Importação de CosFace ajustada para src.models.heads
2. Carregamento de dataset local para evitar erro de Rank/Shape
"""

import sys
import os
import argparse
import tensorflow as tf
import pandas as pd
import numpy as np
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
# Mantemos os imports, mas NÃO usaremos o get_train_val_datasets quebrado
from src.data_loader.face_datasets import LFWValidationCallback 
from src.backbones.resnet import create_resnet50_cosface
from src.optimizers.scheduler import CosineAnnealingScheduler
from src.utils.face_processing import process_image_pipeline_with_detection_flag

# --- CORREÇÃO DO IMPORT AQUI ---
# Antes: from src.losses.margin_losses import CosFace (Isso não existe mais no seu repo)
# Agora:
from src.models.heads import CosFace 
# -------------------------------


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

# -----------------------------------------------------------------------------
# FUNÇÕES DE CARREGAMENTO DE DATASET (LOCAL E SEGURO)
# Substitui o get_train_val_datasets para evitar o erro de Rank/Shape
# -----------------------------------------------------------------------------
def _ensure_image_size_tuple(image_size):
    if isinstance(image_size, (int, float)):
        return (int(image_size), int(image_size))
    elif isinstance(image_size, (tuple, list)) and len(image_size) == 2:
        return (int(image_size[0]), int(image_size[1]))
    else:
        return image_size

def load_dataset_local(dataset_path, config, use_retinaface=False):
    """
    Carrega o dataset de forma segura, evitando o erro de 'Pack' do TensorFlow.
    """
    image_size = _ensure_image_size_tuple(config.image_size)
    batch_size = config.batch_size
    
    train_dir = os.path.join(dataset_path, "train")
    val_dir = os.path.join(dataset_path, "val")
    mapping_csv = os.path.join(dataset_path, "mapping_val.csv")
    
    # Detecção da estrutura
    has_split = os.path.exists(train_dir)
    source_dir = train_dir if has_split else dataset_path
    
    print(f"Carregando treino de: {source_dir}")
    
    # 1. Carregar Dataset de Treino (RAW)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        source_dir,
        validation_split=None if has_split else 0.1,
        subset="training" if not has_split else None,
        seed=123,
        image_size=image_size,
        batch_size=batch_size,
        label_mode='categorical'
    )
    
    class_names = train_ds.class_names
    num_classes = len(class_names)
    
    # 2. Pipeline de Aumentação e Formatação para CosFace
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.1),
    ])

    def prepare_train_batch(images, labels):
        # Aumentação
        images = data_augmentation(images, training=True)
        # Pré-processamento ResNet
        images = tf.keras.applications.resnet50.preprocess_input(images)
        # Formato ((img, label), label)
        return (images, labels), labels

    # Aplicar o map. O segredo é que images e labels JÁ SÃO BATCHES aqui.
    # O TF não vai tentar empacotá-los se a função retornar a estrutura correta.
    train_dataset = train_ds.map(prepare_train_batch, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

    # 3. Carregar Dataset de Validação
    val_dataset = None
    
    # Opção A: Usar Mapping CSV
    if os.path.exists(mapping_csv):
        print("Usando mapping_val.csv para validação.")
        val_dataset, nc_val = load_validation_from_mapping(mapping_csv, dataset_path, image_size, batch_size, class_names, use_retinaface)
        num_classes = max(num_classes, nc_val)
    
    # Opção B: Usar pasta 'val' ou split
    else:
        if has_split and os.path.exists(val_dir):
            print("Usando pasta 'val' para validação.")
            if use_retinaface:
                # RetinaFace precisa de imagens sem batch para filtrar uma a uma, ou batch=1
                val_ds_raw = tf.keras.utils.image_dataset_from_directory(
                    val_dir, seed=123, image_size=image_size, batch_size=None, label_mode='categorical'
                )
                val_dataset = apply_retinaface_validation_filter(val_ds_raw, batch_size)
            else:
                val_ds_raw = tf.keras.utils.image_dataset_from_directory(
                    val_dir, seed=123, image_size=image_size, batch_size=batch_size, label_mode='categorical'
                )
                # Apenas preprocessamento, sem aumentação
                val_dataset = val_ds_raw.map(
                    lambda x, y: ((tf.keras.applications.resnet50.preprocess_input(x), y), y),
                    num_parallel_calls=tf.data.AUTOTUNE
                ).prefetch(tf.data.AUTOTUNE)
        else:
            print("Usando split automático para validação.")
            if use_retinaface:
                val_ds_raw = tf.keras.utils.image_dataset_from_directory(
                    dataset_path, validation_split=0.1, subset="validation", seed=123,
                    image_size=image_size, batch_size=None, label_mode='categorical'
                )
                val_dataset = apply_retinaface_validation_filter(val_ds_raw, batch_size)
            else:
                val_ds_raw = tf.keras.utils.image_dataset_from_directory(
                    dataset_path, validation_split=0.1, subset="validation", seed=123,
                    image_size=image_size, batch_size=batch_size, label_mode='categorical'
                )
                val_dataset = val_ds_raw.map(
                    lambda x, y: ((tf.keras.applications.resnet50.preprocess_input(x), y), y),
                    num_parallel_calls=tf.data.AUTOTUNE
                ).prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset, num_classes

# -----------------------------------------------------------------------------
# UTILS E ESTRATÉGIAS DO SEU CÓDIGO ORIGINAL
# -----------------------------------------------------------------------------

def get_backbone_layers(model):
    try:
        backbone_wrapper = model.get_layer("resnet50_backbone")
        return [layer.name for layer in backbone_wrapper.layers]
    except (ValueError, AttributeError):
        backbone_layer_names = []
        for layer in model.layers:
            if layer.name == 'global_pool': break
            if layer.name != 'label_input': backbone_layer_names.append(layer.name)
        return backbone_layer_names

def get_backbone_layer_objects(model):
    try:
        backbone_wrapper = model.get_layer("resnet50_backbone")
        return backbone_wrapper.layers
    except (ValueError, AttributeError):
        backbone_names = get_backbone_layers(model)
        return [model.get_layer(name) for name in backbone_names]

def strategy_1_full_finetuning(model, config, train_dataset, val_dataset, output_dir, epochs):
    print("\n" + "="*60)
    print("ESTRATÉGIA 1: FINE-TUNING COMPLETO")
    print("="*60)
    
    backbone_layers = get_backbone_layer_objects(model)
    for layer in backbone_layers:
        layer.trainable = True
    
    optimizer = Adam(learning_rate=config.learning_rate)
    crossentropy = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    metrics = [tf.keras.metrics.CategoricalAccuracy(name="accuracy")]
    model.compile(loss=crossentropy, optimizer=optimizer, metrics=metrics)
    
    print(f"Learning rate inicial: {optimizer.learning_rate.numpy():.6f}")
    return model, optimizer

def strategy_2_partial_finetuning(model, config, train_dataset, val_dataset, output_dir, epochs, num_layers=10):
    print("\n" + "="*60)
    print("ESTRATÉGIA 2: FINE-TUNING PARCIAL")
    print("="*60)
    print(f"Apenas as últimas {num_layers} camadas do backbone serão treinadas.")
    
    backbone_layers = get_backbone_layer_objects(model)
    total_layers = len(backbone_layers)
    trainable_start = max(0, total_layers - num_layers)
    
    for layer in backbone_layers:
        layer.trainable = False
    for i in range(trainable_start, total_layers):
        backbone_layers[i].trainable = True
    for layer in model.layers:
        if 'embedding' in layer.name or 'cosface' in layer.name:
            layer.trainable = True
    
    optimizer = Adam(learning_rate=config.learning_rate * 0.1)
    crossentropy = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    metrics = [tf.keras.metrics.CategoricalAccuracy(name="accuracy")]
    model.compile(loss=crossentropy, optimizer=optimizer, metrics=metrics)
    
    print(f"Learning rate inicial: {optimizer.learning_rate.numpy():.6f}")
    return model, optimizer

def strategy_3_differential_lr_finetuning(model, config, train_dataset, val_dataset, output_dir, epochs):
    print("\n" + "="*60)
    print("ESTRATÉGIA 3: FINE-TUNING COM LEARNING RATE DIFERENCIADO")
    print("="*60)
    
    backbone_layers = get_backbone_layer_objects(model)
    for layer in backbone_layers: layer.trainable = True
    
    base_lr = config.learning_rate * 0.1
    optimizer = Adam(learning_rate=base_lr)
    crossentropy = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    metrics = [tf.keras.metrics.CategoricalAccuracy(name="accuracy")]
    model.compile(loss=crossentropy, optimizer=optimizer, metrics=metrics)
    
    print(f"Learning rate médio usado: {optimizer.learning_rate.numpy():.6f}")
    return model, optimizer

def apply_retinaface_validation_filter(val_dataset_raw, batch_size):
    print("\n" + "="*60)
    print("APLICANDO RETINAFACE NA VALIDAÇÃO")
    print("="*60)
    
    def tf_align_with_flag(image, label):
        aligned_img, detection_flag = tf.numpy_function(
            func=process_image_pipeline_with_detection_flag,
            inp=[image],
            Tout=[tf.float32, tf.float32]
        )
        aligned_img.set_shape((112, 112, 3))
        detection_flag.set_shape(())
        return aligned_img, label, detection_flag
    
    def filter_by_detection(aligned_img, label, detection_flag):
        return tf.greater(detection_flag, 0.5)
    
    def remove_flag(aligned_img, label, detection_flag):
        return aligned_img, label
    
    ds = val_dataset_raw.map(tf_align_with_flag, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.filter(filter_by_detection)
    ds = ds.map(remove_flag, num_parallel_calls=tf.data.AUTOTUNE)
    
    ds = ds.map(
        lambda x, y: (tf.keras.applications.resnet50.preprocess_input(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    ds = ds.map(lambda x, y: ((x, y), y), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    print("Filtro RetinaFace aplicado com sucesso.")
    return ds

def plot_finetuning_history(log_path, save_path, strategy_name):
    if not os.path.exists(log_path): return
    log_data = pd.read_csv(log_path)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Acurácia
    axes[0, 0].plot(log_data['epoch'], log_data['accuracy'], label='Treino', marker='o')
    if 'val_accuracy' in log_data.columns:
        axes[0, 0].plot(log_data['epoch'], log_data['val_accuracy'], label='Validação', marker='s')
    axes[0, 0].set_title(f'Acurácia - {strategy_name}')
    axes[0, 0].legend(); axes[0, 0].grid(True)
    
    # Loss
    axes[0, 1].plot(log_data['epoch'], log_data['loss'], label='Treino', marker='o')
    if 'val_loss' in log_data.columns:
        axes[0, 1].plot(log_data['epoch'], log_data['val_loss'], label='Validação', marker='s')
    axes[0, 1].set_title(f'Loss - {strategy_name}')
    axes[0, 1].legend(); axes[0, 1].grid(True)
    
    # Validação LFW
    if 'val_lfw_accuracy' in log_data.columns:
        axes[1, 0].plot(log_data['epoch'], log_data['val_lfw_accuracy'], label='LFW Acurácia', marker='^', color='purple')
        axes[1, 0].set_title('LFW Validação')
        axes[1, 0].legend(); axes[1, 0].grid(True)
    else:
        axes[1, 0].axis('off')
        
    # Gap
    if 'val_accuracy' in log_data.columns:
        axes[1, 1].plot(log_data['epoch'], log_data['accuracy'], label='Treino')
        axes[1, 1].plot(log_data['epoch'], log_data['val_accuracy'], label='Validação')
        axes[1, 1].fill_between(log_data['epoch'], log_data['accuracy'], log_data['val_accuracy'], alpha=0.3)
        axes[1, 1].set_title('Gap Treino-Validação')
        axes[1, 1].legend(); axes[1, 1].grid(True)
    else:
        axes[1, 1].axis('off')

    plt.tight_layout()
    figure_path = os.path.join(save_path, f'finetuning_history_{strategy_name.lower().replace(" ", "_")}.png')
    plt.savefig(figure_path)
    plt.close()

def load_validation_from_mapping(mapping_csv_path, dataset_base_path, image_size, batch_size, class_names, use_retinaface=False):
    print(f"\nCarregando validação do arquivo: {mapping_csv_path}")
    df = pd.read_csv(mapping_csv_path)
    df_val = df[df['split'] == 'val'].copy()
    
    if len(df_val) == 0: raise ValueError(f"Nenhuma amostra de validação no CSV")
    
    unique_ids = sorted(df_val['id'].unique())
    id_to_index = {id_name: idx for idx, id_name in enumerate(unique_ids)}
    
    if class_names is not None:
        missing_ids = set(unique_ids) - set(class_names)
        if missing_ids:
            for id_name in sorted(missing_ids):
                if id_name not in class_names: class_names.append(id_name)
        id_to_index = {id_name: idx for idx, id_name in enumerate(class_names)}
    
    image_paths = []
    labels = []
    for _, row in df_val.iterrows():
        path = row['caminho_imagem']
        if not os.path.isabs(path): path = os.path.join(dataset_base_path, path)
        if os.path.exists(path):
            image_paths.append(path)
            labels.append(id_to_index[row['id']])
    
    num_classes = len(class_names) if class_names is not None else len(unique_ids)
    print(f"Valid: {len(image_paths)} imagens, {len(unique_ids)} classes.")
    
    def load_image_raw(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.cast(img, tf.float32)
        return img, label
        
    def load_image_resized(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, image_size)
        img = tf.cast(img, tf.float32)
        return img, label

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    
    if use_retinaface:
        dataset = dataset.map(load_image_raw, num_parallel_calls=tf.data.AUTOTUNE)
        # O filtro de RetinaFace é aplicado externamente pois lida com batching
        return apply_retinaface_validation_filter(dataset, batch_size), num_classes
    else:
        dataset = dataset.map(load_image_resized, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(lambda x, y: (x, tf.one_hot(y, depth=num_classes)), num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(lambda x, y: ((tf.keras.applications.resnet50.preprocess_input(x), y), y), num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset, num_classes

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def run_finetuning(strategy, pretrained_model_path, dataset_path, output_dir, epochs=30, 
                   num_layers=10, batch_size=64, learning_rate=None, use_retinaface=False, 
                   lfw_path=None, lfw_pairs_path=None):
    
    print("\n" + "="*80)
    print("INICIANDO FINE-TUNING PARA FACE RECOGNITION")
    print("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    log_dir = os.path.join(output_dir, "logs")
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    config = FaceRecognitionConfig()
    config.batch_size = batch_size
    if learning_rate is not None: config.learning_rate = learning_rate
    config.epochs = epochs
    
    # --------------------------------------------------------------------------------
    # CORREÇÃO DO ERRO DE DATASET: CARREGAMENTO LOCAL SEGURO
    # --------------------------------------------------------------------------------
    print(f"\nCarregando dataset de: {dataset_path}")
    image_size_tuple = _ensure_image_size_tuple(config.image_size)
    
    train_path = os.path.join(dataset_path, "train")
    val_path = os.path.join(dataset_path, "val")
    mapping_val_path = os.path.join(dataset_path, "mapping_val.csv")
    
    has_split = os.path.exists(train_path) and os.path.exists(val_path)
    train_source = train_path if has_split else dataset_path
    
    # 1. Carrega RAW (apenas img, label)
    train_dataset_raw = tf.keras.utils.image_dataset_from_directory(
        train_source,
        seed=123,
        image_size=image_size_tuple,
        batch_size=config.batch_size,
        label_mode='categorical',
        validation_split=None if has_split else 0.1,
        subset='training' if not has_split else None
    )
    
    num_classes = len(train_dataset_raw.class_names)
    class_names = train_dataset_raw.class_names
    
    # 2. Pipeline de Validação
    val_dataset = None
    if os.path.exists(mapping_val_path):
        val_dataset, num_classes_val = load_validation_from_mapping(mapping_val_path, dataset_path, image_size_tuple, config.batch_size, class_names, use_retinaface)
        num_classes = max(num_classes, num_classes_val)
    elif has_split:
        if use_retinaface:
             val_raw_nb = tf.keras.utils.image_dataset_from_directory(
                val_path, seed=123, image_size=image_size_tuple, batch_size=None, label_mode='categorical')
             val_dataset = apply_retinaface_validation_filter(val_raw_nb, config.batch_size)
        else:
             val_dataset = tf.keras.utils.image_dataset_from_directory(
                val_path, seed=123, image_size=image_size_tuple, batch_size=config.batch_size, label_mode='categorical')
             val_dataset = val_dataset.map(lambda x, y: ((tf.keras.applications.resnet50.preprocess_input(x), y), y), num_parallel_calls=tf.data.AUTOTUNE)
    else:
        # Split automático
        if use_retinaface:
             val_raw_nb = tf.keras.utils.image_dataset_from_directory(
                dataset_path, validation_split=0.1, subset="validation", seed=123, image_size=image_size_tuple, batch_size=None, label_mode='categorical')
             val_dataset = apply_retinaface_validation_filter(val_raw_nb, config.batch_size)
        else:
             val_dataset = tf.keras.utils.image_dataset_from_directory(
                dataset_path, validation_split=0.1, subset="validation", seed=123, image_size=image_size_tuple, batch_size=config.batch_size, label_mode='categorical')
             val_dataset = val_dataset.map(lambda x, y: ((tf.keras.applications.resnet50.preprocess_input(x), y), y), num_parallel_calls=tf.data.AUTOTUNE)

    # 3. Pipeline de Treino SEGURO (Substitui os .maps antigos que causavam Pack error)
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.1),
    ])
    
    def prepare_train_batch(images, labels):
        # AQUI está a correção: processamos o batch de uma vez e retornamos a tupla aninhada explicitamente
        # Isso evita que o TF tente "empacotar" tensores de ranks diferentes
        images = data_augmentation(images, training=True)
        images = tf.keras.applications.resnet50.preprocess_input(images)
        return (images, labels), labels

    train_dataset = train_dataset_raw.map(prepare_train_batch, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    
    config.update_num_classes(num_classes)
    print(f"Número de classes no dataset: {config.num_classes}")
    
    # --------------------------------------------------------------------------------
    # FIM DA CORREÇÃO DO DATASET - RESTO DO CÓDIGO PERMANECE IDÊNTICO
    # --------------------------------------------------------------------------------
    
    pretrained_model = load_pretrained_model(pretrained_model_path, config)
    
    cosface_layer = pretrained_model.get_layer("cosface_loss")
    if cosface_layer.n_classes != config.num_classes:
        print(f"\nAjustando número de classes de {cosface_layer.n_classes} para {config.num_classes}")
        model, feature_extractor, new_backbone = create_resnet50_cosface(config)
        try:
            try:
                pretrained_backbone = pretrained_model.get_layer("resnet50_backbone")
                new_backbone.set_weights(pretrained_backbone.get_weights())
                print("Pesos do backbone transferidos com sucesso.")
            except (ValueError, AttributeError):
                pretrained_backbone_layers = get_backbone_layer_objects(pretrained_model)
                new_backbone_layers = new_backbone.layers
                for pretrained_layer, new_layer in zip(pretrained_backbone_layers, new_backbone_layers):
                    if pretrained_layer.get_weights(): new_layer.set_weights(pretrained_layer.get_weights())
                print("Pesos do backbone transferidos (camada a camada).")
            
            try:
                pretrained_embedding = pretrained_model.get_layer("embedding_dense")
                new_embedding = model.get_layer("embedding_dense")
                if pretrained_embedding.get_weights()[0].shape[0] == new_embedding.get_weights()[0].shape[0]:
                    new_embedding.set_weights(pretrained_embedding.get_weights())
                    print("Pesos de embedding transferidos.")
            except: pass
        except Exception as e:
            print(f"Aviso transferência: {e}")
    else:
        model = pretrained_model
    
    # Estratégias
    if strategy == '1':
        model, optimizer = strategy_1_full_finetuning(model, config, train_dataset, val_dataset, output_dir, epochs)
    elif strategy == '2':
        model, optimizer = strategy_2_partial_finetuning(model, config, train_dataset, val_dataset, output_dir, epochs, num_layers)
    elif strategy == '3':
        model, optimizer = strategy_3_differential_lr_finetuning(model, config, train_dataset, val_dataset, output_dir, epochs)
    else:
        raise ValueError(f"Estratégia inválida: {strategy}")
    
    # Callbacks
    checkpoint_path = os.path.join(checkpoint_dir, f"finetuning_epoch_{{epoch:02d}}.keras")
    log_path = os.path.join(log_dir, f"finetuning_log.csv")
    
    callbacks = [
        ModelCheckpoint(filepath=checkpoint_path, monitor='loss', mode='min', verbose=1, save_best_only=False),
        CSVLogger(log_path, append=False),
        CosineAnnealingScheduler(epochs, float(optimizer.learning_rate.numpy()), 1e-6)
    ]
    
    # Validação LFW (Segura)
    if lfw_path and lfw_pairs_path and os.path.exists(lfw_path):
        print(f"\nConfigurando validação LFW...")
        try:
            inp = model.input[0] 
            # Recriar output embedding seguro
            x = model.get_layer("resnet50_backbone")(inp)
            x = model.get_layer("global_pool")(x)
            x = model.get_layer("embedding_bn")(x)
            x = model.get_layer("embedding_dense")(x)
            out = model.get_layer("embedding_dense_bn")(x)
            
            feature_extractor_for_val = tf.keras.Model(inputs=inp, outputs=out)
            
            # Carregar pares manualmente (sem depender da função removida)
            pairs = []
            with open(lfw_pairs_path, 'r') as f:
                for line in f.readlines()[1:]: pairs.append(line.strip().split())
            
            callbacks.append(LFWValidationCallback(
                feature_extractor=feature_extractor_for_val,
                lfw_path=lfw_path,
                pairs_path=lfw_pairs_path, # Passamos o path pois o callback carrega internamente
                image_size=image_size_tuple
            ))
            print("Callback LFW adicionado.")
        except Exception as e:
            print(f"ERRO LFW: {e}. Continuando sem.")
            
    if val_dataset is not None:
        callbacks.append(EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1))
    
    # Treinar
    print(f"\n{'='*80}\nIniciando treinamento\n{'='*80}\n")
    history = model.fit(
        x=train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    final_model_path = os.path.join(output_dir, f"final_model.keras")
    model.save(final_model_path)
    print(f"\nModelo final salvo em: {final_model_path}")
    plot_finetuning_history(log_path, figures_dir, strategy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tuning Face Rec")
    parser.add_argument('--strategy', type=str, required=True, choices=['1', '2', '3'])
    parser.add_argument('--pretrained_model', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--num_layers', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--use_retinaface', action='store_true')
    parser.add_argument('--lfw_path', type=str, default=None)
    parser.add_argument('--lfw_pairs_path', type=str, default=None)
    
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
        use_retinaface=args.use_retinaface,
        lfw_path=args.lfw_path,
        lfw_pairs_path=args.lfw_pairs_path
    )