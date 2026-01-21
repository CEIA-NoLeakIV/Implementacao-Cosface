# Local: src/models/builder.py
import tensorflow as tf
from src.models.backbones import build_resnet50_backbone
from src.models.heads import CosFace

def build_face_model(config):
    """
    Monta o modelo completo de Reconhecimento Facial:
    Input Image + Input Label -> Backbone -> CosFace Layer -> Softmax Output
    """
    # 1. Construir Backbone
    backbone = build_resnet50_backbone(
        input_shape=config.image_size + (3,) if isinstance(config.image_size, tuple) else (config.image_size, config.image_size, 3),
        embedding_size=512
    )
    
    # 2. Definir Inputs do Modelo Final
    input_image = backbone.input
    input_label = tf.keras.layers.Input(shape=(config.num_classes,), name="input_label") # One-hot esperado
    
    # 3. Obter Embedding
    embedding = backbone(input_image)
    
    # 4. Aplicar Camada CosFace
    # Importante: O nome 'cosface_loss' ajuda na identificação posterior para validação
    cosface_layer = CosFace(n_classes=config.num_classes, 
                            s=30.0, 
                            m=0.35, 
                            name="cosface_loss")
    
    output = cosface_layer([embedding, input_label])
    
    # 5. Saída Final (Softmax)
    # CosFace retorna logits, então aplicamos softmax para calcular a loss padrão
    output = tf.keras.layers.Activation('softmax', name="predictions")(output)
    
    # 6. Criar Modelo Keras
    model = tf.keras.models.Model(inputs=[input_image, input_label], outputs=output, name="cosface_model")
    
    return model
