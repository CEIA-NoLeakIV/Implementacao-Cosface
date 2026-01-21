# Mantenha em Cosface/src/models/heads.py
import tensorflow as tf
import math

class CosFace(tf.keras.layers.Layer):
    def __init__(self, n_classes, s=30.0, m=0.35, regularizer=None, **kwargs):
        super(CosFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = tf.keras.regularizers.get(regularizer)

    def build(self, input_shape):
        embedding_shape = input_shape[0]
        # Inicializador 'glorot_normal' ou 'he_normal' ajuda na convergência inicial
        self.w = self.add_weight(name='weights',
                                 shape=(embedding_shape[-1], self.n_classes),
                                 initializer='glorot_normal', 
                                 trainable=True,
                                 regularizer=self.regularizer)
        super(CosFace, self).build(input_shape)

    def call(self, inputs):
        embedding, label = inputs
        
        embedding_norm = tf.nn.l2_normalize(embedding, axis=1)
        w_norm = tf.nn.l2_normalize(self.w, axis=0)
        
        cosine = tf.matmul(embedding_norm, w_norm)

        phi = cosine - self.m
        
        phi = tf.clip_by_value(phi, -1.0, 1.0)
        
        if label.shape[-1] != self.n_classes:
            label = tf.one_hot(tf.cast(label, tf.int32), self.n_classes)
        
        label = tf.cast(label, cosine.dtype)
   
        output = (label * phi) + ((1.0 - label) * cosine)
        
        return output * self.s
