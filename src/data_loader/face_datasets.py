# Arquivo: src/data_loader/face_datasets.py
import os
import tensorflow as tf
from tensorflow.keras import layers
# Importamos a função que criamos acima
from src.utils.face_processing import process_image_pipeline

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
], name="data_augmentation")

def get_train_val_datasets(path, image_size, batch_size, validation_split=0.1, align_faces=False):
    """
    Args:
        align_faces (bool): Se True, ativa a detecção e alinhamento MTCNN.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset não encontrado: {path}")

    # 1. Carregar Dataset (usando estrutura de pastas padrão)
    # Nota: image_size aqui é inicial, o alinhamento corrigirá depois
    train_ds = tf.keras.utils.image_dataset_from_directory(
        path, validation_split=validation_split, subset="training", 
        seed=123, image_size=(image_size, image_size), batch_size=batch_size, label_mode='categorical'
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        path, validation_split=validation_split, subset="validation", 
        seed=123, image_size=(image_size, image_size), batch_size=batch_size, label_mode='categorical'
    )
    
    # Capturar número de classes
    num_classes = len(train_ds.class_names)

    # 2. Wrapper para o TensorFlow chamar o Python/OpenCV
    def tf_align_wrapper(image, label):
        # A mágica acontece aqui: tf.numpy_function executa código python arbitrário
        aligned_img = tf.numpy_function(
            func=process_image_pipeline,
            inp=[image],
            Tout=tf.float32
        )
        # É crucial definir o shape manual depois do numpy_function
        aligned_img.set_shape((112, 112, 3)) 
        return aligned_img, label

    # 3. Pipeline de Transformação
    def apply_transform(ds, is_train=False):
        # A. Alinhamento (Se ativado) - DEVE ser o primeiro passo
        if align_faces:
            ds = ds.map(tf_align_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
        
        # B. Augmentation (Apenas treino)
        if is_train:
            ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), 
                        num_parallel_calls=tf.data.AUTOTUNE)
            
        # C. Pré-processamento ResNet (Normalização de cores)
        ds = ds.map(lambda x, y: (tf.keras.applications.resnet50.preprocess_input(x), y), 
                    num_parallel_calls=tf.data.AUTOTUNE)
        
        return ds.prefetch(tf.data.AUTOTUNE)

    # Aplicar aos datasets
    train_ds = apply_transform(train_ds, is_train=True)
    val_ds = apply_transform(val_ds, is_train=False)

    return train_ds, val_ds, num_classes
