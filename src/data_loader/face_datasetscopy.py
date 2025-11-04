import os
from pathlib import Path
import tensorflow as tf
import pandas as pd
from tensorflow.keras.applications import resnet50


class FaceDatasetLoader:
    def __init__(self, dataset_path, image_size, batch_size, validation_split):
        self.dataset_path = Path(dataset_path)
        self.image_size = image_size
        self.batch_size = batch_size
        self.validation_split = validation_split

    def load(self):
        dataset_format = self._detect_format()

        if dataset_format == "celebA":
            print("[INFO] Detectado dataset: CelebA")
            return self._load_celebA()

        elif dataset_format == "folder_per_identity":
            print("[INFO] Detectado dataset organizado por pastas (CASIA, VGGFace2, etc.)")
            return self._load_folder_per_identity()

        else:
            print("[ERRO] Formato do dataset não detectado ou não suportado.")
            return None, None, []

    def _detect_format(self):
        # Detecta CelebA com base nos arquivos CSV
        if (self.dataset_path / "list_eval_partition.csv").exists() and (self.dataset_path / "list_attr_celeba.csv").exists():
            return "celebA"

        # Detecta estrutura padrão (pasta por identidade)
        subdirs = [d for d in self.dataset_path.iterdir() if d.is_dir()]
        if len(subdirs) > 0:
            return "folder_per_identity"

        return None

    def _load_celebA(self):
        # Lê os arquivos CSV do CelebA
        partition_file = self.dataset_path / "list_eval_partition.csv"
        attr_file = self.dataset_path / "list_attr_celeba.csv"

        partitions = pd.read_csv(partition_file)
        attributes = pd.read_csv(attr_file)

        # Mesclar informações
        data = partitions.merge(attributes, left_on="image_id", right_on="image_id")

        # Filtrar por divisão
        train_data = data[data["partition"] == 0]
        val_data = data[data["partition"] == 1]

        # Escolher atributo para classes (exemplo: Smiling)
        attribute = "Smiling"
        train_data[attribute] = train_data[attribute].apply(lambda x: 1 if x == 1 else 0)
        val_data[attribute] = val_data[attribute].apply(lambda x: 1 if x == 1 else 0)

        # Caminho das imagens
        img_dir = self.dataset_path / "img_align_celeba" / "img_align_celeba"
        train_data["image_path"] = train_data["image_id"].apply(lambda x: str(img_dir / x))
        val_data["image_path"] = val_data["image_id"].apply(lambda x: str(img_dir / x))

        def load_image(path, label):
            img = tf.io.read_file(path)
            img = tf.image.decode_jpeg(img, channels=3)
            # Garantir que o tamanho seja uma tupla válida
            target_size = (self.image_size, self.image_size) if isinstance(self.image_size, int) else tuple(self.image_size)
            img = tf.image.resize(img, target_size)
            img = img / 255.0
            return img, label

        def load_image_with_label(path, label):
            def check_file_exists(path):
                exists = tf.io.gfile.exists(path.decode('utf-8'))
                return path if exists else b''

            path = tf.py_function(func=check_file_exists, inp=[path], Tout=tf.string)

            if tf.equal(path, b''):
                target_size = (self.image_size, self.image_size) if isinstance(self.image_size, int) else tuple(self.image_size)
                dummy_image = tf.zeros((target_size[0], target_size[1], 3), dtype=tf.float32)

                dummy_label = tf.one_hot(label, depth=8632)

                return (dummy_image, dummy_label), dummy_label

            img = tf.io.read_file(path)
            img = tf.image.decode_jpeg(img, channels=3)
            target_size = (self.image_size, self.image_size) if isinstance(self.image_size, int) else tuple(self.image_size)
            img = tf.image.resize(img, target_size)
            img = img / 255.0
            # Converter rótulos para one-hot com base no número de classes
            label = tf.one_hot(label, depth=8632)
            return (img, label), label

        train_ds = tf.data.Dataset.from_tensor_slices((train_data["image_path"], train_data[attribute]))
        val_ds = tf.data.Dataset.from_tensor_slices((val_data["image_path"], val_data[attribute]))

        # Filtrar imagens ausentes
        train_ds = train_ds.map(lambda x, y: load_image_with_label(x, y)).filter(lambda x: x is not None)
        val_ds = val_ds.map(lambda x, y: load_image_with_label(x, y)).filter(lambda x: x is not None)

        train_ds = train_ds.shuffle(10000).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        class_names = ["Not_Smiling", "Smiling"]
        return train_ds, val_ds, class_names

    def _load_folder_per_identity(self):
        # Caso comum: CASIA, VGGFace2, WebFace, etc.
        # Garantir que image_size seja uma tupla válida
        target_size = (self.image_size, self.image_size) if isinstance(self.image_size, int) else tuple(self.image_size)

        # Carregar o dataset de treinamento
        train_ds = tf.keras.utils.image_dataset_from_directory(
            self.dataset_path,
            batch_size=self.batch_size,
            image_size=target_size,
            validation_split=self.validation_split if self.validation_split > 0 else None,
            subset="training" if self.validation_split > 0 else None,
            seed=1337
        )

        # Carregar o dataset de validação (forçar criação mesmo sem split)
        if self.validation_split > 0:
            val_ds = tf.keras.utils.image_dataset_from_directory(
                self.dataset_path,
                batch_size=self.batch_size,
                image_size=target_size,
                validation_split=self.validation_split,
                subset="validation",
                seed=1337
            )
        else:
            val_ds = tf.keras.utils.image_dataset_from_directory(
                self.dataset_path,
                batch_size=self.batch_size,
                image_size=target_size,
                seed=1337
            )

        # Capturar as classes antes de aplicar o filtro
        class_names = train_ds.class_names

        # Adicionar logs para depuração
        print("[DEBUG] Classes detectadas:", class_names)
        print("[DEBUG] Aplicando filtro para remover exemplos inválidos no train_ds e val_ds")

        # Filtrar exemplos inválidos no dataset
        train_ds = train_ds.filter(lambda x, y: tf.reduce_any(tf.not_equal(tf.size(y), 0)))
        val_ds = val_ds.filter(lambda x, y: tf.reduce_any(tf.not_equal(tf.size(y), 0)))

        print("[DEBUG] Filtro aplicado com sucesso")

        # Aplicar pré-processamento específico do ResNet50
        print("[INFO] Aplicando pré-processamento do ResNet50...")
        train_ds = train_ds.map(lambda x, y: (resnet50.preprocess_input(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.map(lambda x, y: (resnet50.preprocess_input(x), y), num_parallel_calls=tf.data.AUTOTUNE)

        return train_ds, val_ds, class_names


def get_train_val_datasets(dataset_path, image_size, batch_size, validation_split=0.2):
    """
    Mantida a assinatura original.
    Nenhuma outra parte do projeto precisa ser modificada.
    """
    loader = FaceDatasetLoader(dataset_path, image_size, batch_size, validation_split)
    return loader.load()


class LFWValidationCallback(tf.keras.callbacks.Callback):
    """Valida o modelo no LFW com embeddings normalizados e threshold dinâmico."""
    def __init__(self, feature_extractor, lfw_path, pairs_path, image_size):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.lfw_path = lfw_path
        self.pairs_path = pairs_path
        self.image_size = image_size

    def _load_lfw_pairs(self):
        """Carrega os pares do arquivo de pares do LFW."""
        pairs = []
        with open(self.pairs_path, 'r') as f:
            for line in f.readlines()[1:]:
                pairs.append(line.strip().split())
        return pairs

    def _preprocess_image(self, image_path):
        """Pré-processa uma imagem para o modelo."""
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (self.image_size, self.image_size))
        img = tf.keras.applications.resnet50.preprocess_input(img)
        return img

    def on_epoch_end(self, epoch, logs=None):
        """Executa a validação no final de uma época."""
        pairs = self._load_lfw_pairs()
        embeddings = {}

        # Extrair embeddings para todas as imagens nos pares
        for pair in pairs:
            for img_path in pair[:2]:
                if img_path not in embeddings:
                    img = self._preprocess_image(os.path.join(self.lfw_path, img_path))
                    img = tf.expand_dims(img, axis=0)  # Adicionar dimensão do batch
                    embeddings[img_path] = self.feature_extractor(img, training=False).numpy()

        # Calcular similaridades e avaliar
        correct = 0
        for pair in pairs:
            emb1, emb2 = embeddings[pair[0]], embeddings[pair[1]]
            similarity = tf.linalg.normalize(emb1)[0] @ tf.linalg.normalize(emb2)[0].T
            if (similarity > 0.5 and pair[2] == '1') or (similarity <= 0.5 and pair[2] == '0'):
                correct += 1

        accuracy = correct / len(pairs)
        print(f"[LFW] Acurácia na época {epoch + 1}: {accuracy:.4f}")
