# -*- coding: utf-8 -*-
"""Position embedding layers.
"""
from keras.engine.base_layer import Layer
from keras import backend as K


class PositionEmbedding(Layer):

    def __init__(self, method='sum', embedding_dim=None, **kwargs):
        """
        # 此层Layer仅可放在Embedding之后。
        # 参数：
        #    - embedding_dim: position_embedding的维度，为None或者偶数（Google给的PositionEmbedding构造公式分奇偶数）；
        #    - method: word_embedding与position_embedding的结合方法，求和sum或拼接concatenate；
        #        -- sum: position_embedding的值与word_embedding相加，需要将embedding_dim定义得和word_embedding一样；默认方式，FaceBook的论文和Google论文中用的都是后者；
        #        -- concatenate：将position_embedding的值拼接在word_embedding后面。
        """
        self.method = method
        self.embedding_dim = embedding_dim
        super(PositionEmbedding, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        if self.method == 'sum':
            return input_shape
        elif self.method == 'concatenate':
            return (input_shape[0], input_shape[1], input_shape[2] + self.embedding_dim)
        else:
            raise TypeError('Method not understood:', self.method)

    def call(self, word_embeddings):
        """
        # 参照keras.engine.base_layer的call方法。
        # 将word_embeddings中的第p个词语映射为一个d_pos维的position_embedding，其中第i个元素的值为PE_i(p)，计算公式如下：
        #     PE_2i(p) = sin(p/10000^(2i/d_pos))
        #     PE_2i+1(p) = cos(p/10000^(2i/d_pos))
        # 参数
        #     - word_embeddings: Tensor or list/tuple of tensors.
        # 返回
        #     - position_embeddings：Tensor or list/tuple of tensors.
        """
        if (self.embedding_dim == None) or (self.method == 'sum'):
            self.embedding_dim = int(word_embeddings.shape[-1])
        batch_size, sequence_length = K.shape(word_embeddings)[0], K.shape(word_embeddings)[1]
        # 生成(self.embedding_dim,)向量：1/(10000^(2*[0,1,2,...,self.embedding_dim-1]/self.embedding_dim))，对应公式中的1/10000^(2i/d_pos)
        embedding_wise_pos = 1. / K.pow(10000., 2 * K.arange(self.embedding_dim / 2,
                                                             dtype='float32') / self.embedding_dim)  # n_dims=1, shape=(self.embedding_dim,)
        # 增加维度
        embedding_wise_pos = K.expand_dims(embedding_wise_pos, 0)  # n_dims=2, shape=(1,self.embedding_dim)
        # 生成(batch_size,sequence_length,)向量，基础值为1，首层为0，按层累加(第一层值为0，第二层值为1，...)，对应公式中的p
        word_wise_pos = K.cumsum(K.ones_like(word_embeddings[:, :, 0]),
                                 axis=1) - 1  # n_dims=2, shape=(batch_size,sequence_length)
        # 增加维度
        word_wise_pos = K.expand_dims(word_wise_pos, 2)  # n_dims=3, shape=(batch_size,sequence_length,1)
        # 生成(batch_size,sequence_length,self.embedding_dim)向量，对应公式中的p/10000^(2i/d_pos)
        position_embeddings = K.dot(word_wise_pos, embedding_wise_pos)
        # 直接concatenate无法出现交替现象，应先升维再concatenate再reshape
        position_embeddings = K.reshape(
            K.concatenate([K.cos(position_embeddings), K.sin(position_embeddings)], axis=-1),
            shape=(batch_size, sequence_length, -1))
        if self.method == 'sum':
            return word_embeddings + position_embeddings
        elif self.method == 'concatenate':
            return K.concatenate([word_embeddings, position_embeddings], axis=-1)