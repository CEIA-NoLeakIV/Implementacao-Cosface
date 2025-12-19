from __future__ import annotations
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import (
    Input,
    Dense,
    GlobalAveragePooling2D,
    BatchNormalization,
    Dropout,
)
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
import tensorflow as tf

# CORREÇÃO AQUI: Importação ajustada para usar o caminho relativo correto a partir da raiz 'src'
from src.models.heads import CosFace

def _build_classification_head(config, backbone_output):
    """Create the projection head that feeds the CosFace layer."""

    x = GlobalAveragePooling2D(name="global_pool")(backbone_output)
    x = BatchNormalization(name="embedding_bn")(x)
    x = Dropout(0.4, name="embedding_dropout")(x)
    x = Dense(
        config.embedding_size,
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(1e-4),
        name="embedding_dense",
    )(x)
    x = BatchNormalization(name="embedding_dense_bn")(x)
    return x

print(">>> EXECUTANDO ARQUIVO DE MODELO CORRETO <<<")

def create_resnet50_cosface(config):
    """Create the ResNet50 backbone followed by a CosFace classification head."""

    # O próprio modelo ResNet50 pode ser tratado como uma camada e nomeado.
    backbone = ResNet50(include_top=False, weights="imagenet", input_shape=config.input_shape, name="resnet50_backbone")
    backbone.trainable = False

    input_image = backbone.input
    num_classes = getattr(config, "num_classes", None)
    if num_classes is None:
        num_classes = getattr(config, "NUM_CLASSES")
    input_label = Input(shape=(num_classes,), name="label_input")

    embeddings = _build_classification_head(config, backbone.output)

    cosface_logits = CosFace(
        num_classes,
        s=config.cosine_scale,
        m=config.cosine_margin,
        name="cosface_loss",
    )([embeddings, input_label])

    model = Model(inputs=[input_image, input_label], outputs=cosface_logits, name="resnet50_cosface")
    feature_extractor = Model(inputs=input_image, outputs=embeddings, name="resnet50_feature_extractor")

    return model, feature_extractor, backbone

__all__ = ["create_resnet50_cosface"]
