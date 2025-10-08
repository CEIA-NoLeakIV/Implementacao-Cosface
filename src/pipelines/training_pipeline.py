import os
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping

from src.backbones.resnet import create_resnet50_cosface
from src.data_loader.face_datasets import get_train_val_datasets, LFWValidationCallback
from src.optimizers.scheduler import CosineAnnealingScheduler


def plot_training_history(log_data, save_path):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(log_data['epoch'], log_data['accuracy'], label='Treino Acurácia')
    ax1.plot(log_data['epoch'], log_data['val_accuracy'], label='Validação Acurácia')
    ax1.set_title('Histórico de Acurácia (VGGFace2)')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Acurácia')
    ax1.legend()
    ax2.plot(log_data['epoch'], log_data['loss'], label='Treino Perda')
    ax2.plot(log_data['epoch'], log_data['val_loss'], label='Validação Perda')
    ax2.set_title('Histórico de Perda (VGGFace2)')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Perda')
    ax2.legend()
    fig.tight_layout()
    plt.savefig(os.path.join(save_path, 'face_training_history.png'))
    print(f"Gráfico do histórico de treino salvo em: {save_path}")


def plot_learning_rate(log_data, save_path):
    if 'lr' in log_data.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(log_data['epoch'], log_data['lr'], label='Taxa de Aprendizagem')
        plt.title('Agendamento da Taxa de Aprendizagem')
        plt.xlabel('Época')
        plt.ylabel('Learning Rate')
        plt.legend()
        plt.savefig(os.path.join(save_path, 'face_learning_rate.png'))
        print(f"Gráfico da taxa de aprendizado salvo em: {save_path}")


def run_face_training(config, vggface2_path, lfw_path, lfw_pairs_path):
    print("--- Iniciando Pipeline de Treinamento de Faces ---")
    FIGURES_PATH = 'reports/figures'
    os.makedirs(FIGURES_PATH, exist_ok=True)
    os.makedirs(os.path.dirname(config.CHECKPOINT_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(config.LOG_PATH), exist_ok=True)

    # Carrega dados
    train_dataset, val_dataset, num_classes_in_subset = get_train_val_datasets(
        vggface2_path, config.IMAGE_SIZE, config.BATCH_SIZE, fraction=config.DATASET_FRACTION
    )
    config.NUM_CLASSES = num_classes_in_subset

    # Cria modelo
    model, feature_extractor = create_resnet50_cosface(config)

    # Fase 1: treina apenas a cabeça
    print("\n--- FASE 1: Treinando apenas a cabeça do modelo ---")
    optimizer_head = Adam(learning_rate=1e-3)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
    model.compile(loss=loss_fn, optimizer=optimizer_head, metrics=['accuracy'])
    model.fit(train_dataset.take(2000),
              epochs=5,
              validation_data=val_dataset.take(200),
              verbose=1)

    # Fase 2: fine‑tuning
    print("\n--- FASE 2: Iniciando fine-tuning do modelo completo ---")
    optimizer_finetune = SGD(learning_rate=config.LEARNING_RATE, momentum=config.MOMENTUM)
    model.compile(loss=loss_fn, optimizer=optimizer_finetune, metrics=['accuracy'])
    print("Sumário do modelo para fine-tuning:")
    model.summary()

    # Cosine annealing scheduler
    scheduler = CosineAnnealingScheduler(T_max=config.EPOCHS,
                                         eta_max=config.LEARNING_RATE,
                                         eta_min=config.MIN_LEARNING_RATE,
                                         verbose=1)

    # Callbacks
    lfw_callback = LFWValidationCallback(feature_extractor, lfw_path, lfw_pairs_path, config.IMAGE_SIZE)
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5, mode='min', restore_best_weights=True)
    checkpoint_path_h5 = config.CHECKPOINT_PATH.replace('.keras', '.h5')
    weights_checkpoint_path_h5 = checkpoint_path_h5.replace('.h5', '.weights.h5')

    callbacks = [
        ModelCheckpoint(checkpoint_path_h5, monitor='val_loss', mode='min', verbose=1, save_best_only=True),
        ModelCheckpoint(weights_checkpoint_path_h5, monitor='val_loss', mode='min', verbose=1, save_best_only=True, save_weights_only=True),
        CSVLogger(config.LOG_PATH),
        lfw_callback,
        scheduler,
        early_stopping_callback,
    ]

    history = model.fit(train_dataset,
                        epochs=config.EPOCHS,
                        callbacks=callbacks,
                        validation_data=val_dataset,
                        verbose=1)

    print("\n--- Treinamento Concluído ---")
    print("\n--- Iniciando Análise Pós-Treino ---")
    if os.path.exists(config.LOG_PATH):
        log_data = pd.read_csv(config.LOG_PATH)
        plot_training_history(log_data, FIGURES_PATH)
        plot_learning_rate(log_data, FIGURES_PATH)
    else:
        print(f"AVISO: Arquivo de log não encontrado em {config.LOG_PATH}. Pulando geração de gráficos.")
