# -*- coding: utf-8 -*-
"""K-MaxPooling layers.
"""
from keras.engine.base_layer import Layer, InputSpec
from keras import backend as K
import tensorflow as tf


class KMaxPooling1D(Layer):
    """
    k-max-pooling
    """

    def __init__(self, k=1, **kwargs):
        super(KMaxPooling1D, self).__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)
        self.k = k

    def compute_output_shape(self, input_shape):
        return (input_shape[0], (input_shape[2] * self.k))

    def call(self, inputs):
        # swap last two dimensions since top_k will be applied along the last dimension
        shifted_input = tf.transpose(inputs, [0, 2, 1])
        # extract top_k, returns two tensors [values, indices]
        top_k = tf.nn.top_k(shifted_input, k=self.k, sorted=True, name=None)[0]
        return top_k