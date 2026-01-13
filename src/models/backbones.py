import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation, Input, MaxPooling2D, Flatten, GlobalAveragePooling2D
from tensorflow.keras import regularizers

def build_resnet50_backbone(input_shape=(112, 112, 3), embedding_size=512):
    """Backbone padrão ResNet50 conforme configurado no nosso treino estável."""
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(embedding_size, name="embedding_layer")(x)
    return Model(inputs=base_model.input, outputs=x, name="resnet50_backbone")

def vgg_block(x, filters, layers, weight_decay=1e-4):
    for _ in range(layers):
        x = Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    return x

def build_vgg8_backbone(input_shape=(28, 28, 1), embedding_size=512):
    input_layer = Input(shape=input_shape)
    x = vgg_block(input_layer, 16, 2)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = vgg_block(x, 32, 2)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = vgg_block(x, 64, 2)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(embedding_size, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    return Model(inputs=input_layer, outputs=x, name="vgg8_backbone")
