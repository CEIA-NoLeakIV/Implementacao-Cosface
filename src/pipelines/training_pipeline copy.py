import os
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping

from src.backbones.resnet import create_resnet50_cosface
from src.data_loader.face_datasets import get_train_val_datasets
from src.optimizers.scheduler import CosineAnnealingScheduler
from src.losses.margin_losses import CosFace


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


def run_face_training(config, dataset_path, resume_training=False):
    print("--- Iniciando Pipeline de Treinamento de Faces ---")
    checkpoint_path = os.path.join("experiments", "Resnet50_vgg_cropado_CelebA", "checkpoints", "epoch_{epoch:02d}.keras")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    log_path = os.path.join("experiments", "Resnet50_vgg_cropado_CelebA", "logs", "training_log.csv")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    figures_path = os.path.join("experiments", "Resnet50_vgg_cropado_CelebA", "figures")
    os.makedirs(figures_path, exist_ok=True)

    train_dataset, _, num_classes_in_subset = get_train_val_datasets(
        dataset_path,
        config.image_size,
        config.batch_size,
    )
    config.update_num_classes(num_classes_in_subset)
    print(f"Número de classes no subset: {config.num_classes}")

    # Mapear o dataset para o formato de entrada e alvo esperado pelo modelo
    train_dataset = train_dataset.map(
        lambda image, label: ((image, tf.one_hot(label, depth=config.num_classes)), tf.one_hot(label, depth=config.num_classes)),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    initial_epoch = 0
    model = None

    if resume_training:
        print("\n--- TENTANDO RESUMIR TREINAMENTO ---")
        if os.path.exists(checkpoint_path):
            print(f"Checkpoint encontrado em {checkpoint_path}. Carregando modelo...")
            model = tf.keras.models.load_model(
                checkpoint_path,
                custom_objects={'CosFace': CosFace}
            )
            
            if os.path.exists(log_path):
                log_data = pd.read_csv(log_path)
                if not log_data.empty:
                    initial_epoch = log_data['epoch'].iloc[-1] + 1
                print(f"Resumindo treinamento a partir da época {initial_epoch}.")
            else:
                print("AVISO: Arquivo de log não encontrado. Não foi possível determinar a época inicial. Começando da época 0.")
        else:
            print(f"AVISO: Nenhum checkpoint encontrado em {checkpoint_path}. Iniciando um novo treinamento do zero.")
            resume_training = False

    if model is None:
        model, _, backbone = create_resnet50_cosface(config)
    else:
        backbone = model.get_layer("resnet50_backbone")
    cosface_layer = model.get_layer("cosface_loss")
    original_margin = float(cosface_layer.m)
    original_scale = float(cosface_layer.s)

    if not resume_training:
        print("\n--- FASE 1: Treinando apenas a cabeça do modelo ---")
        backbone.trainable = False
        warmup_scale = min(10.0, original_scale)
        cosface_layer.m = 0.0
        cosface_layer.s = warmup_scale

        optimizer_head = Adam(learning_rate=1e-3)
        crossentropy = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        metrics = [tf.keras.metrics.CategoricalAccuracy(name="accuracy")]
        model.compile(loss=crossentropy, optimizer=optimizer_head, metrics=metrics)
        model.fit(
            train_dataset,
            epochs=config.warmup_epochs,
            verbose=1,
        )

        print("\n--- FASE 1.5: Adaptando a cabeça do modelo à margem ---")
        cosface_layer.m = original_margin
        cosface_layer.s = original_scale
        optimizer_margin_warmup = Adam(learning_rate=1e-4)
        model.compile(loss=crossentropy, optimizer=optimizer_margin_warmup, metrics=metrics)
        model.fit(
            train_dataset,
            epochs=3,
            verbose=1,
        )

    print("\n--- FASE 2: Iniciando fine-tuning do modelo completo ---")
    backbone.trainable = True
    if config.trainable_backbone_layers > 0:
        for layer in backbone.layers[:-config.trainable_backbone_layers]:
            layer.trainable = False
    
    cosface_layer.m = original_margin
    cosface_layer.s = original_scale

    if not resume_training:
        fine_tuning_lr = 1e-4
        print(f"Configurando fine-tuning com learning rate: {fine_tuning_lr}")
        optimizer_finetune = Adam(learning_rate=fine_tuning_lr)
        crossentropy = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        metrics = [tf.keras.metrics.CategoricalAccuracy(name="accuracy")]
        model.compile(loss=crossentropy, optimizer=optimizer_finetune, metrics=metrics)
    
    print("Sumário do modelo para fine-tuning:")
    model.summary()

    current_lr = model.optimizer.learning_rate.numpy()
    scheduler = CosineAnnealingScheduler(T_max=config.epochs, eta_max=current_lr, eta_min=config.min_learning_rate, verbose=1)
    
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='loss',
        mode='min',
        verbose=1,
        save_best_only=False,
        save_weights_only=False
    )

    callbacks = [
        scheduler,
        checkpoint_callback,
        CSVLogger(log_path, append=True),
    ]

    history = model.fit(train_dataset,
                        epochs=config.epochs,
                        callbacks=callbacks,
                        initial_epoch=initial_epoch,
                        verbose=1)

    print("\n--- Treinamento Concluído ---")
