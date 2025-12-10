# Arquivo: src/pipelines/training_pipeline.py
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
    
    if 'accuracy' in log_data.columns:
        ax1.plot(log_data['epoch'], log_data['accuracy'], label='Treino Acurácia')
    if 'val_accuracy' in log_data.columns:
        ax1.plot(log_data['epoch'], log_data['val_accuracy'], label='Validação Acurácia')
    
    ax1.set_title('Histórico de Acurácia')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Acurácia')
    ax1.legend()
    
    if 'loss' in log_data.columns:
        ax2.plot(log_data['epoch'], log_data['loss'], label='Treino Perda')
    if 'val_loss' in log_data.columns:
        ax2.plot(log_data['epoch'], log_data['val_loss'], label='Validação Perda')
        
    ax2.set_title('Histórico de Perda')
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


def run_face_training(config, dataset_path, resume_training=False, align_faces=False):
    """
    Executa o pipeline de treinamento.
    
    Args:
        config: Objeto de configuração.
        dataset_path: Caminho para as imagens.
        resume_training: Se True, tenta carregar checkpoint existente.
        align_faces: Se True, ativa o alinhamento facial (MTCNN/UniFace) no carregamento.
    """
    print("--- Iniciando Pipeline de Treinamento de Faces ---")
    print(f"--- Modo de Alinhamento Facial: {'ATIVADO' if align_faces else 'DESATIVADO'} ---")

    # Caminhos para salvar artefatos
    checkpoint_path = os.path.join("experiments", "Resnet50_vgg_cropado_CelebA", "checkpoints", "epoch_{epoch:02d}.keras")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    log_path = os.path.join("experiments", "Resnet50_vgg_cropado_CelebA", "logs", "training_log.csv")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    figures_path = os.path.join("experiments", "Resnet50_vgg_cropado_CelebA", "figures")
    os.makedirs(figures_path, exist_ok=True)

    # Carregar Dataset
    # Passamos o argumento align_faces para o loader atualizado
    train_dataset, _, num_classes_in_subset = get_train_val_datasets(
        dataset_path,
        config.image_size,
        config.batch_size,
        fraction=config.dataset_fraction,
        align_faces=align_faces  # <--- NOVA FLAG AQUI
    )
    
    config.update_num_classes(num_classes_in_subset)
    print(f"Número de classes no subset: {config.num_classes}")

    initial_epoch = 0
    model = None

    # Lógica de Resume Training
    if resume_training:
        print("\n--- TENTANDO RESUMIR TREINAMENTO ---")
        # Procura o último checkpoint (lógica simplificada, ideal seria buscar o maior epoch)
        # Se você tiver um caminho fixo para o "last.keras", use-o aqui.
        # Caso contrário, o keras load_model precisa de um arquivo específico.
        # Aqui assumimos que se o usuário pediu resume, ele verificou se existe algo.
        # Para robustez, vamos procurar arquivos na pasta.
        ckpt_dir = os.path.dirname(checkpoint_path)
        if os.path.exists(ckpt_dir) and len(os.listdir(ckpt_dir)) > 0:
            # Pega o arquivo mais recente ou específico se implementado
            # Por enquanto, mantemos a lógica original mas verificamos se existe algo
            # Nota: O código original procurava um checkpoint específico "epoch_{epoch:02d}.keras" 
            # o que não funciona bem para load sem formatar. Vamos assumir que o usuário quer continuar
            # do último log ou de um arquivo específico se a lógica fosse mais complexa.
            
            # Se você tiver um arquivo fixo "last_model.keras", use:
            last_ckpt = os.path.join(ckpt_dir, "last_model.keras") 
            
            if os.path.exists(last_ckpt): # Exemplo, ajuste conforme seu salvamento
                print(f"Carregando modelo de: {last_ckpt}")
                model = tf.keras.models.load_model(last_ckpt, custom_objects={'CosFace': CosFace})
            else:
                print("AVISO: Nenhum arquivo 'last_model.keras' encontrado. Tentando criar novo.")
                resume_training = False
            
            if os.path.exists(log_path) and resume_training:
                try:
                    log_data = pd.read_csv(log_path)
                    if not log_data.empty:
                        initial_epoch = log_data['epoch'].iloc[-1] + 1
                    print(f"Resumindo treinamento a partir da época {initial_epoch}.")
                except Exception as e:
                    print(f"Erro ao ler log: {e}. Começando da época 0.")
        else:
            print(f"AVISO: Pasta de checkpoints vazia. Iniciando do zero.")
            resume_training = False

    # Criação do Modelo (se não carregado)
    if model is None:
        model, _, backbone = create_resnet50_cosface(config)
    else:
        backbone = model.get_layer("resnet50_backbone")
    
    cosface_layer = model.get_layer("cosface_loss")
    original_margin = float(cosface_layer.m)
    original_scale = float(cosface_layer.s)

    # --- FASE 1: Warmup (Apenas se não for resume) ---
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

    # --- FASE 2: Fine-tuning Completo ---
    print("\n--- FASE 2: Iniciando fine-tuning do modelo completo ---")
    backbone.trainable = True
    if config.trainable_backbone_layers > 0:
        for layer in backbone.layers[:-config.trainable_backbone_layers]:
            layer.trainable = False
    
    cosface_layer.m = original_margin
    cosface_layer.s = original_scale

    # Recompilar se mudou a trainabilidade ou se é novo treino
    # Se for resume, o load_model já traz o optimizer, mas recompilar garante LR correto
    fine_tuning_lr = 1e-4
    if not resume_training:
        print(f"Configurando fine-tuning com learning rate inicial: {fine_tuning_lr}")
        optimizer_finetune = Adam(learning_rate=fine_tuning_lr)
    else:
        # Se resume, tentamos manter o otimizador carregado ou criar novo com LR baixo
        print(f"Continuando fine-tuning.")
        optimizer_finetune = Adam(learning_rate=fine_tuning_lr) # Reinicia o otimizador para garantir

    crossentropy = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    metrics = [tf.keras.metrics.CategoricalAccuracy(name="accuracy")]
    model.compile(loss=crossentropy, optimizer=optimizer_finetune, metrics=metrics)
    
    print("Sumário do modelo para fine-tuning:")
    model.summary()

    current_lr = float(model.optimizer.learning_rate.numpy()) if hasattr(model.optimizer.learning_rate, 'numpy') else fine_tuning_lr
    
    # Callbacks
    scheduler = CosineAnnealingScheduler(T_max=config.epochs, eta_max=current_lr, eta_min=config.min_learning_rate, verbose=1)
    
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='loss',
        mode='min',
        verbose=1,
        save_best_only=False, # Salva todas as épocas conforme string de formatação
        save_weights_only=False
    )
    
    # Callback extra para salvar sempre o "last_model.keras" para facilitar resume
    last_ckpt_path = os.path.join(os.path.dirname(checkpoint_path), "last_model.keras")
    last_checkpoint_callback = ModelCheckpoint(
        filepath=last_ckpt_path,
        save_best_only=False,
        save_weights_only=False,
        verbose=0
    )

    callbacks = [
        scheduler,
        checkpoint_callback,
        last_checkpoint_callback,
        CSVLogger(log_path, append=True),
    ]

    history = model.fit(train_dataset,
                        epochs=config.epochs,
                        callbacks=callbacks,
                        initial_epoch=initial_epoch,
                        verbose=1)

    # Plotar resultados finais
    if os.path.exists(log_path):
        try:
            log_data = pd.read_csv(log_path)
            plot_training_history(log_data, figures_path)
            plot_learning_rate(log_data, figures_path)
        except Exception as e:
            print(f"Erro ao gerar gráficos finais: {e}")

    print("\n--- Treinamento Concluído ---")
