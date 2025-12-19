import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Lightweight augmentation pipeline reused across training batches.
data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
    ],
    name="data_augmentation",
)

def filter_valid_faces(directory):
    """
    Lógica simplificada para garantir que estamos treinando com rostos.
    Em uma implementação real, você chamaria um detector aqui.
    Para o framework ficar leve, esta função pode apenas verificar integridade 
    ou ser expandida com um detector como MTCNN.
    """
    print(f"Filtrando imagens em: {directory}...")
    # Aqui entraria a lógica de detecção. 
    # Por ora, vamos garantir que ela retorne o caminho validado.
    return directory 

def get_train_val_datasets(path, image_size, batch_size, validation_split=0.02, align_faces=False, fraction=1.0):
    # Se a flag estiver ativa, filtramos o caminho ou os arquivos antes de prosseguir
    if align_faces:
        path = filter_valid_faces(path)
    
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Dataset não encontrado: {path}")
    
    if not os.path.isdir(path):
        data_root = os.environ.get("DATA_ROOT", "<não definido>")
        raise FileNotFoundError(
            f"Diretório do dataset não encontrado: {path}. "
            f"Defina DATA_ROOT apontando para a pasta que contém 'raw/vggface2_112x112'. "
            f"DATA_ROOT atual: {data_root}"
        )

    # Lógica para detectar se o dataset é pré-dividido
    train_path = os.path.join(path, "train")
    val_path = os.path.join(path, "val")
    
    pre_split = os.path.isdir(train_path) and os.path.isdir(val_path)

    if pre_split:
        print(f"Dataset pré-dividido detectado. Usando '{train_path}' para treino e '{val_path}' para validação.")
        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_path,
            labels='inferred',
            label_mode='categorical',
            image_size=image_size,
            interpolation='bilinear',
            batch_size=batch_size,
            shuffle=True,
            seed=123
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            val_path,
            labels='inferred',
            label_mode='categorical',
            image_size=image_size,
            interpolation='bilinear',
            batch_size=batch_size,
            shuffle=False, # Validação não precisa de shuffle
            seed=123
        )
        # O número de classes é o número de pastas no diretório de treino
        num_classes_to_use = len(os.listdir(train_path))

    else:
        print(f"Dataset não-dividido detectado. Criando divisão de {validation_split*100}% a partir de '{path}'.")
        # Mantém o comportamento antigo de criar a divisão
        train_ds = tf.keras.utils.image_dataset_from_directory(
            path,
            labels='inferred',
            label_mode='categorical',
            image_size=image_size,
            interpolation='bilinear',
            batch_size=batch_size,
            shuffle=True,
            seed=123,
            validation_split=validation_split,
            subset='training'
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            path,
            labels='inferred',
            label_mode='categorical',
            image_size=image_size,
            interpolation='bilinear',
            batch_size=batch_size,
            shuffle=False,
            seed=123,
            validation_split=validation_split,
            subset='validation'
        )
        num_classes_to_use = len(os.listdir(path))

    # As funções de pré-processamento e o resto do pipeline permanecem os mesmos
    def augment_and_prepare(image, label):
        one_hot_label = label # O label já vem como one-hot de image_dataset_from_directory
        image = data_augmentation(image, training=True)
        image = tf.keras.applications.resnet50.preprocess_input(image)
        # Retornar features como lista [image, label] para evitar ambiguidade no Keras
        return [image, one_hot_label], one_hot_label

    def validate_and_prepare(image, label):
        one_hot_label = label # O label já vem como one-hot
        image = tf.keras.applications.resnet50.preprocess_input(image)
        return [image, one_hot_label], one_hot_label

    train_ds = train_ds.map(augment_and_prepare, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(validate_and_prepare, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, num_classes_to_use


class LFWValidationCallback(tf.keras.callbacks.Callback):
    """Valida o modelo no LFW com embeddings normalizados e threshold dinâmico."""
    def __init__(self, feature_extractor, lfw_path, pairs_path, image_size):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.lfw_path = lfw_path
        self.image_size = image_size
        self.pairs = self._read_pairs(pairs_path)

    def _read_pairs(self, pairs_filename):
        pairs = []
        with open(pairs_filename, 'r') as f:
            for line in f.readlines()[1:]:
                pairs.append(line.strip().split())
        return pairs

    def _preprocess_image(self, file_path):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, self.image_size)
        # mesmo preprocess_input do treino
        img = tf.keras.applications.resnet50.preprocess_input(img)
        return img

    def on_epoch_end(self, epoch, logs=None):
        print("\nIniciando validação no LFW...")
        embeddings1 = []
        embeddings2 = []
        actual_issame = []

        for pair in self.pairs:
            if len(pair) == 3:
                path1 = os.path.join(self.lfw_path, pair[0], f"{pair[0]}_{int(pair[1]):04d}.jpg")
                path2 = os.path.join(self.lfw_path, pair[0], f"{pair[0]}_{int(pair[2]):04d}.jpg")
                actual_issame.append(True)
            else:
                path1 = os.path.join(self.lfw_path, pair[0], f"{pair[0]}_{int(pair[1]):04d}.jpg")
                path2 = os.path.join(self.lfw_path, pair[2], f"{pair[2]}_{int(pair[3]):04d}.jpg")
                actual_issame.append(False)

            img1 = self._preprocess_image(path1)
            img2 = self._preprocess_image(path2)

            emb1 = self.feature_extractor(tf.expand_dims(img1, axis=0), training=False)
            emb2 = self.feature_extractor(tf.expand_dims(img2, axis=0), training=False)

            # normaliza embeddings
            emb1 = tf.nn.l2_normalize(emb1, axis=1)
            emb2 = tf.nn.l2_normalize(emb2, axis=1)

            embeddings1.append(emb1)
            embeddings2.append(emb2)

        emb1_tensor = tf.concat(embeddings1, axis=0)
        emb2_tensor = tf.concat(embeddings2, axis=0)
        similarities = tf.reduce_sum(tf.multiply(emb1_tensor, emb2_tensor), axis=1).numpy()

        # encontra o melhor threshold
        best_acc = 0.0
        best_thr = 0.0
        thresholds = np.linspace(-1.0, 1.0, 401)
        for thr in thresholds:
            preds = similarities > thr
            acc = np.mean(preds == np.array(actual_issame))
            if acc > best_acc:
                best_acc = acc
                best_thr = thr
        print(f"LFW Validação - Melhor threshold: {best_thr:.3f}, Acurácia: {best_acc:.4f}")
        if logs is not None:
            logs['val_lfw_accuracy'] = best_acc
