import tensorflow as tf
from tensorflow.keras.layers import Input, Activation
from tensorflow.keras.models import Model
from .backbones import build_resnet50_backbone
from .heads import CosFace

def build_face_model(config):
    """Constrói o modelo completo integrando backbone e cabeça CosFace."""
    image_input = Input(shape=(*config.image_size, 3), name="image_input")
    label_input = Input(shape=(config.num_classes,), name="label_input")

    # 1. Extrator de características (Backbone)
    backbone = build_resnet50_backbone(
        input_shape=(*config.image_size, 3), 
        embedding_size=512
    )
    embedding = backbone(image_input)

    # 2. Cabeça CosFace
    # Note que passamos o label como entrada para a camada durante o treino
    logits = CosFace(n_classes=config.num_classes, name="cosface_loss")([embedding, label_input])
    
    # 3. Saída com Softmax
    output = Activation('softmax', name="cosine_softmax")(logits)

    model = Model(inputs=[image_input, label_input], outputs=output)
    return model
