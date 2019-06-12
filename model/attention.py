# -*- coding: utf-8 -*-
"""Attention layers.
"""
from keras.engine.base_layer import Layer
from keras import backend as K


class Attention(Layer):

    def __init__(self, multiheads, head_dim, mask_right=False, **kwargs):
        """
        # 参数：
        #    - multiheads: Attention的数目
        #    - head_dim: Attention Score的维度
        #    - mask_right: Position-wise Mask，在Encoder时不使用，在Decoder时使用
        """
        self.multiheads = multiheads
        self.head_dim = head_dim
        self.output_dim = multiheads * head_dim
        self.mask_right = mask_right
        super(Attention, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1],
                self.output_dim)  # shape=[batch_size,Q_sequence_length,self.multiheads*self.head_dim]

    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ',
                                  shape=(input_shape[0][-1], self.output_dim),  # input_shape[0] -> Q_seq
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK',
                                  shape=(input_shape[1][-1], self.output_dim),  # input_shape[1] -> K_seq
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV',
                                  shape=(input_shape[2][-1], self.output_dim),  # input_shape[2] -> V_seq
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(Attention, self).build(input_shape)

    def Mask(self, inputs, seq_len, mode='add'):
        """
        # 需要对sequence进行Mask以忽略填充部分的影响，一般将填充部分设置为0。
        # 由于Attention中的Mask要放在softmax之前，则需要给softmax层输入一个非常大的负整数，以接近0。
        # 参数：
        #    - inputs: 输入待mask的sequence
        #    - seq_len: shape=[batch_size,1]或[batch_size,]
        #    - mode: mask的方式，'mul'时返回的mask位置为0，'add'时返回的mask位置为一个非常大的负数，在softmax下为0。由于attention的mask是在softmax之前，所以要用这种方式执行
        """
        if seq_len == None:
            return inputs
        else:
            # seq_len[:,0].shape=[batch_size,1]
            # short_sequence_length=K.shape(inputs)[1]：较短的sequence_length，如K_sequence_length，V_sequence_length
            mask = K.one_hot(indices=seq_len[:, 0], num_classes=K.shape(inputs)[
                1])  # mask.shape=[batch_size,short_sequence_length],mask=[[0,0,0,0,1,0,0,..],[0,1,0,0,0,0,0...]...]
            mask = 1 - K.cumsum(mask,
                                axis=1)  # mask.shape=[batch_size,short_sequence_length],mask=[[1,1,1,1,0,0,0,...],[1,0,0,0,0,0,0,...]...]
            # 将mask增加到和inputs一样的维度，目前仅有两维[0],[1]，需要在[2]上增加维度
            for _ in range(len(inputs.shape) - 2):
                mask = K.expand_dims(mask, 2)
            # mask.shape=[batch_size,short_sequence_length,1,1]
            if mode == 'mul':
                # Element-wise multiply：直接做按位与操作
                # return_shape = inputs.shape
                # 返回值：[[seq_element_1,seq_element_2,...,masked_1,masked_2,...],...]，其中seq_element_i,masked_i的维度均为2维
                # masked_i的值为0
                return inputs * mask
            elif mode == 'add':
                # Element-wise add：直接做按位加操作
                # return_shape = inputs.shape
                # 返回值：[[seq_element_1,seq_element_2,...,masked_1,masked_2,...],...]，其中seq_element_i,masked_i的维度均为2维
                # masked_i的值为一个非常大的负数，在softmax下为0。由于attention的mask是在softmax之前，所以要用这种方式执行
                return inputs - (1 - mask) * 1e12

    def call(self, QKVs):
        """
        # 参照keras.engine.base_layer的call方法。
        # 1. Q',K',V' = Q .* WQ_i,K .* WK_i,V .* WV_i
        # 2. head_i = Attention(Q',K',V') = softmax((Q' .* K'.T)/sqrt(d_k)) .* V
        # 3. MultiHead(Q,K,V) = Concat(head_1,...,head_n)
        # 参数
            - QKVs：[Q_seq,K_seq,V_seq]或[Q_seq,K_seq,V_seq,Q_len,V_len]
                -- Q_seq.shape = [batch_size,Q_sequence_length,Q_embedding_dim]
                -- K_seq.shape = [batch_size,K_sequence_length,K_embedding_dim]
                -- V_seq.shape = [batch_size,V_sequence_length,V_embedding_dim]
                -- Q_len.shape = [batch_size,1],如：[[7],[5],[3],...]
                -- V_len.shape = [batch_size,1],如：[[7],[5],[3],...]
        # 返回
            -
        """
        # 如果只传入Q_seq,K_seq,V_seq，那么就不做Mask
        # 如果同时传入Q_seq,K_seq,V_seq,Q_len,V_len，那么对多余部分做Mask
        if len(QKVs) == 3:
            Q_seq, K_seq, V_seq = QKVs
            Q_len, V_len = None, None
        elif len(QKVs) == 5:
            Q_seq, K_seq, V_seq, Q_len, V_len = QKVs
        # 对Q、K、V做线性变换，以Q为例进行说明
        # Q_seq.shape=[batch_size,Q_sequence_length,Q_embedding_dim]
        # self.WQ.shape=[Q_embedding_dim,self.output_dim]=[Q_embedding_dim,self.multiheads*self.head_dim]
        Q_seq = K.dot(Q_seq,
                      self.WQ)  # Q_seq.shape=[batch_size,Q_sequence_length,self.output_dim]=[batch_size,Q_sequence_length,self.multiheads*self.head_dim]
        Q_seq = K.reshape(Q_seq, shape=(-1, K.shape(Q_seq)[1], self.multiheads,
                                        self.head_dim))  # Q_seq.shape=[batch_size,Q_sequence_length,self.multiheads,self.head_dim]
        Q_seq = K.permute_dimensions(Q_seq, pattern=(
        0, 2, 1, 3))  # Q_seq.shape=[batch_size,self.multiheads,Q_sequence_length,self.head_dim]
        # 对K做线性变换，和Q一样
        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(K_seq, shape=(-1, K.shape(K_seq)[1], self.multiheads, self.head_dim))
        K_seq = K.permute_dimensions(K_seq, pattern=(0, 2, 1, 3))
        # 对V做线性变换，和Q一样
        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(V_seq, shape=(-1, K.shape(V_seq)[1], self.multiheads, self.head_dim))
        V_seq = K.permute_dimensions(V_seq, pattern=(0, 2, 1, 3))
        # 计算内积
        A = K.batch_dot(Q_seq, K_seq, axes=[3, 3]) / K.sqrt(K.cast(self.head_dim,
                                                                   dtype='float32'))  # A.shape=[batch_size,self.multiheads,Q_sequence_length,K_sequence_length]
        A = K.permute_dimensions(A, pattern=(
        0, 3, 2, 1))  # A.shape=[batch_size,K_sequence_length,Q_sequence_length,self.multiheads]
        # Mask部分：
        # 1.Sequence-wise Mask(axis=1)：这部分不是Attention论文提出的操作，而是常规应该有的mask操作（类似于Keras.pad_sequence）
        # 原始输入A的形状，[batch_size,K_sequence_length,Q_sequence_length,self.multiheads]
        # 这部分是为了mask掉sequence的填充部分，比如V_len=5,那么对于A需要在K_sequence_length部分进行mask
        # 这部分不好理解的话可以想象为在句子长度上进行mask，统一对齐到V_len
        A = self.Mask(A, V_len, 'add')
        A = K.permute_dimensions(A, pattern=(
        0, 3, 2, 1))  # A.shape=[batch_size,self.multiheads,Q_sequence_length,K_sequence_length]
        # 2.Position-wise Mask(axis=2)：这部分是Attention论文提出的操作，在Encoder时不使用，在Decoder时使用
        # 原始输入A的形状，[batch_size,self.multiheads,Q_sequence_length,K_sequence_length]
        # 这部分是为了mask掉后续Position的影响，确保Position_i的预测输出仅受Position_0~Position_i的影响
        # 这部分不好理解的话可以想象为为进行实时机器翻译时，机器是无法获取到人后面要说的是什么话，它能获得的信息只能是出现过的词语
        if self.mask_right:
            ones = K.ones_like(A[:1, :1])  # ones.shape=[1,1,Q_sequence_length,K_sequence_length],生成全1矩阵
            lower_triangular = K.tf.matrix_band_part(ones, num_lower=-1,
                                                     num_upper=0)  # lower_triangular.shape=ones.shape，生成下三角阵
            mask = (
                               ones - lower_triangular) * 1e12  # mask.shape=ones.shape，生成类上三角阵（注：这里不能用K.tf.matrix_band_part直接生成上三角阵，因为对角线元素需要丢弃），同样需要乘以一个很大的数（减去这个数）,以便在softmax时趋于0
            A = A - mask  # Element-wise subtract，A.shape=[batch_size,self.multiheads,Q_sequence_length,K_sequence_length]
        A = K.softmax(A)  # A.shape=[batch_size,self.multiheads,Q_sequence_length,K_sequence_length]
        # V_seq.shape=[batch_size,V_sequence_length,V_embedding_dim]
        O_seq = K.batch_dot(A, V_seq,
                            axes=[3, 2])  # O_seq.shape=[batch_size,self.multiheads,Q_sequence_length,V_sequence_length]
        O_seq = K.permute_dimensions(O_seq, pattern=(
        0, 2, 1, 3))  # O_seq.shape=[batch_size,Q_sequence_length,self.multiheads,V_sequence_length]
        # 这里有个坑，维度计算时要注意：(batch_size*V_sequence_length)/self.head_dim要为整数
        O_seq = K.reshape(O_seq, shape=(
        -1, K.shape(O_seq)[1], self.output_dim))  # O_seq.shape=[,Q_sequence_length,self.multiheads*self.head_dim]
        O_seq = self.Mask(O_seq, Q_len, 'mul')
        return O_seq