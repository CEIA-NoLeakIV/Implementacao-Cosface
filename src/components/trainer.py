# Local: src/components/trainer.py
import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, TensorBoard
from src.data_loader.face_datasets import LFWValidationCallback

class Trainer:
    def __init__(self, model, config, output_dir):
        self.model = model
        self.config = config
        self.output_dir = output_dir
        
        # Criar diretórios
        self.checkpoint_dir = os.path.join(output_dir, 'checkpoints')
        self.log_dir = os.path.join(output_dir, 'logs')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    def compile(self, learning_rate=0.01): # Comece com 0.01 em vez de 0.1 se travar em 0
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=10000,
            decay_rate=0.9)
            
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=True)
        
        self.model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy')]
        )

    def fit(self, train_dataset, val_dataset=None, lfw_config=None):
        # Se lfw_config for passado (path do LFW), adicionamos o callback de similaridade
        callbacks = self._get_callbacks()
        if lfw_config:
            # Extrai o backbone do modelo completo para o callback
            backbone = self.model.get_layer("resnet50_backbone") 
            lfw_cb = LFWValidationCallback(
                feature_extractor=backbone,
                lfw_path=lfw_config['path'],
                pairs_path=lfw_config['pairs'],
                image_size=self.config.image_size
            )
            callbacks.append(lfw_cb)
            
        return self.model.fit(train_dataset, validation_data=val_dataset, 
                             epochs=self.config.epochs, callbacks=callbacks)

    def _get_callbacks(self, lfw_config=None):
        # 1. Mantém os callbacks que você já tinha
        ckpt_path = os.path.join(self.checkpoint_dir, "epoch_{epoch:02d}_val_acc_{val_accuracy:.4f}.keras")
        checkpoint = ModelCheckpoint(
            filepath=ckpt_path,
            monitor='val_accuracy',
            save_best_only=False,
            verbose=1
        )
        
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        tensorboard = TensorBoard(log_dir=os.path.join(self.log_dir, 'tensorboard'))
        
        # 2. ADICIONA a nova lógica de persistência de logs (CSV)
        # Isso garante que se o tmux cair, os dados estão no CSV
        csv_logger = CSVLogger(os.path.join(self.log_dir, "training_log.csv"), append=True)
        
        callbacks = [checkpoint, early_stop, tensorboard, csv_logger]
        
        # 3. Integra o LFW se configurado
        if lfw_config:
            from src.data_loader.face_datasets import LFWValidationCallback
            # Pega apenas o backbone para validar similaridade
            backbone = self.model.get_layer("resnet50_backbone") 
            lfw_cb = LFWValidationCallback(
                feature_extractor=backbone,
                lfw_path=lfw_config['path'],
                pairs_path=lfw_config['pairs'],
                image_size=self.config.image_size
            )
            callbacks.append(lfw_cb)
            
        return callbacks