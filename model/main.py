import sys
import os
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import jieba
import re
import matplotlib.pyplot as plt
import keras
from keras import Input
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D, Dense, Flatten, Embedding, Dropout, concatenate, Bidirectional, LSTM, GRU, Reshape
from keras.models import Model
from keras.utils import to_categorical
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras_layer_normalization import LayerNormalization
from keras import backend as K
from keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from position_embedding import PositionEmbedding #自定义层：Position Embedding
from attention import Attention #自定义层：Attention
from k_max_pooling import KMaxPooling1D #自定义层：K-MaxPooling

def resample(data_dict,sample_size,seed,test_rate=0.1):
    """
    数据重采样
    :param data_dict: 输入数据dict类型：{label_1:[text_11,text_12,...,text_1j,...],label_2:[...],...,label_i:[...],...}
    :param sample_size: 每类样本容量：超过的样本类进行下采样，不足的样本类进行上采样
    :param seed: 随机种子
    :param test_rate: 测试集比例
    :return: train：训练集；test：测试集
    """
    random.seed(seed)
    np.random.seed(seed)
    train = []
    test = []
    for label in data_dict:
        #先抽取测试集
        num_data = len(data_dict[label])
        test_num_data = max(int(num_data // (1/test_rate)),1)
        for i in range(test_num_data):
            test_data = np.random.choice(data_dict[label],1)
            test_index = np.where(data_dict[label]==test_data)
            if test_num_data > 1:
                data_dict[label] = np.delete(data_dict[label],test_index)
            test.append([test_data[0],label])
        #再抽取训练集
        if len(data_dict[label]) >= sample_size:
            #下采样(不能重复，要用random.sample)
            train_data = random.sample(list(data_dict[label]),sample_size)
            train.extend([[train_data_i,label] for train_data_i in train_data])
        else:
            #上采样（有重复，要用np.random.choice+原数据）
            train_data = list(data_dict[label])
            train_data_extend = np.random.choice(train_data,sample_size-len(train_data)).tolist()
            train_data.extend(train_data_extend)
            train.extend([[train_data_i,label] for train_data_i in train_data])
    return train,test

regex = re.compile(u'[^\u4E00-\u9FA5|0-9a-zA-Z]')
def remove_punctuation(s):
    """
    文本标准化：仅保留中文字符、英文字符和数字字符
    :param s: 输入文本（中文文本要求进行分词处理）
    :return: s_：标准化的文本
    """
    s_ = regex.sub('', s)
    return s_

def text_tokenizer(texts):
    """
    文本分词+标准化
    :param texts: 输入文本list类型：[text_1,text_2,...,text_i,...]
    :return: 标准化后的分词文本
    """
    return [jieba.lcut(remove_punctuation(text)) for text in texts]


class ClassificationReport(Callback):
    """
    keras callback方法，在每个训练epoch后返回指标计算结果，包含[precision, recall, f1, support]
    """
    def __init__(self, target_names, **kwargs):
        self.target_names = target_names
        super(ClassificationReport, self).__init__(**kwargs)

    def on_train_begin(self, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        val_pred = np.array([item.argmax() for item in self.model.predict(self.validation_data[0])])
        val_true = np.array([item.argmax() for item in self.validation_data[1]])
        print(classification_report(y_true=val_true, y_pred=val_pred, target_names=self.target_names))
        return

def Transformer(train,test,val,output_categories,num_words,maxlen,embedding_dim,callbacks,
                multiheads=8,head_dim=16,dense_units=256,dropout_rate=0.2,
                batch_size=512,epochs=10,verbose=1):
    """
    Transformer-Encoder模型
    :param train: 训练数据，格式：(data,label)
    :param test: 测试数据，格式：(data,label)
    :param val: 验证数据，格式：(data,label)
    :param output_categories: 分类类别数
    :param num_words: 限制top_n词数（文本内仅计算top_n的词向量）
    :param maxlen: 最大句长限制（计算基本单元：词组）
    :param embedding_dim: 词向量维度
    :param callbacks: 每个epoch进行回调计算的指标
    :param multiheads: Attention层的多头数量
    :param head_dim: Attention层的头维度
    :param dense_units: Dense层神经元数量
    :param dropout_rate:
    :param batch_size:
    :param epochs:
    :param verbose:
    :return: model：模型；train_acc：训练精度；val_acc：验证精度；test_acc：测试精度
    """
    #一、数据准备
    train_data,train_label = train
    test_data,test_label = test
    val_data,val_label = val
    #一、网络构建
    inputs = Input(shape=(maxlen,))
    embedding = Embedding(input_dim=num_words,input_length=maxlen,output_dim=embedding_dim,trainable=True)(inputs)
    embedding = PositionEmbedding()(embedding)
    attention = Attention(multiheads=multiheads,head_dim=head_dim,mask_right=False)([embedding,embedding,embedding])
    attention_layer_norm = LayerNormalization()(attention)
    dropout = Dropout(dropout_rate)(attention_layer_norm)
    flatten = Flatten()(dropout)
    dense = Dense(dense_units,activation='relu')(flatten)
    dense_layer_norm = LayerNormalization()(dense)
    outputs = Dense(output_categories,activation='softmax')(dense_layer_norm)
    model = Model(inputs=inputs,outputs=outputs)
    model.summary()
    model.compile(optimizer=keras.optimizers.Adam(lr=0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-8),
              loss='categorical_crossentropy',
              metrics=['acc'])
    #二、模型训练及指标返回
    history = model.fit(train_data,train_label,batch_size=batch_size,validation_data=(val_data,val_label),callbacks=callbacks,epochs=epochs,verbose=verbose)
    train_acc = history.history['acc']
    val_acc = history.history['val_acc']
    test_loss,test_acc = model.evaluate(test_data,test_label)
    return model,train_acc,val_acc,test_acc

def BiGRU(train,test,val,output_categories,num_words,maxlen,embedding_dim,callbacks,
          gru_units=256,dropout=0.2,recurrent_dropout=0.2,
          batch_size=512,epochs=10,verbose=1):
    """
    TextRNN模型：双向GRU
    :param train: 训练数据，格式：(data,label)
    :param test: 测试数据，格式：(data,label)
    :param val: 验证数据，格式：(data,label)
    :param output_categories: 分类类别数
    :param num_words: 限制top_n词数（文本内仅计算top_n的词向量）
    :param maxlen: 最大句长限制（计算基本单元：词组）
    :param embedding_dim: 词向量维度
    :param callbacks: 每个epoch进行回调计算的指标
    :param gru_units: GRU层神经元数量
    :param dropout:
    :param recurrent_dropout:
    :param batch_size:
    :param epochs:
    :param verbose:
    :return: model：模型；train_acc：训练精度；val_acc：验证精度；test_acc：测试精度
    """
    #一、数据准备
    train_data,train_label = train
    test_data,test_label = test
    val_data,val_label = val
    #一、网络构建
    inputs = Input(shape=(maxlen,))
    embedding = Embedding(input_dim=num_words,input_length=maxlen,output_dim=embedding_dim,trainable=True)(inputs)
    bigru = Bidirectional(GRU(units=gru_units,dropout=dropout,recurrent_dropout=recurrent_dropout,return_sequences=True))(embedding)
    bigru = Bidirectional(GRU(units=gru_units,dropout=dropout,recurrent_dropout=recurrent_dropout,activation='relu'))(bigru)
    outputs = Dense(output_categories,activation='softmax')(bigru)
    model = Model(inputs=inputs,outputs=outputs)
    model.summary()
    model.compile(optimizer=keras.optimizers.Adam(lr=0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-8),
              loss='categorical_crossentropy',
              metrics=['acc'])
    #二、模型训练及指标返回
    history = model.fit(train_data,train_label,batch_size=batch_size,validation_data=(val_data,val_label),callbacks=callbacks,epochs=epochs,verbose=verbose)
    train_acc = history.history['acc']
    val_acc = history.history['val_acc']
    test_loss,test_acc = model.evaluate(test_data,test_label)
    return model,train_acc,val_acc,test_acc

def TextCNN(train,test,val,output_categories,num_words,maxlen,embedding_dim,callbacks,
            kernels=[2,3,4],filters=128,dropout_rate=0.5,dense_units=256,
            batch_size=512,epochs=10,verbose=1):
    """
    TextCNN模型：Conv1D + MaxPooling1D
    :param train: 训练数据，格式：(data,label)
    :param test: 测试数据，格式：(data,label)
    :param val: 验证数据，格式：(data,label)
    :param output_categories: 分类类别数
    :param num_words: 限制top_n词数（文本内仅计算top_n的词向量）
    :param maxlen: 最大句长限制（计算基本单元：词组）
    :param embedding_dim: 词向量维度
    :param callbacks: 每个epoch进行回调计算的指标
    :param kernels: 卷积核数量list类型
    :param filters: 卷积层输出维度
    :param dropout_rate:
    :param dense_units: Dense层神经元数量
    :param batch_size:
    :param epochs:
    :param verbose:
    :return: model：模型；train_acc：训练精度；val_acc：验证精度；test_acc：测试精度
    """
    #一、数据准备
    train_data,train_label = train
    test_data,test_label = test
    val_data,val_label = val
    #一、网络构建
    inputs = Input(shape=(maxlen,))
    embedding = Embedding(input_dim=num_words,input_length=maxlen,output_dim=embedding_dim,trainable=True)(inputs)
    pool_output = []
    for kernel_size in kernels:
        conv = Conv1D(filters=filters,kernel_size=kernel_size,strides=1,padding='same',activation='relu')(embedding)
        pooling = MaxPooling1D()(conv)
        pool_output.append(pooling)
    pool_output = concatenate(pool_output)
    dropout = Dropout(dropout_rate)(pool_output)
    flatten = Flatten()(dropout)
    dense = Dense(dense_units,activation='relu')(flatten)
    outputs = Dense(output_categories,activation='softmax')(dense)
    model = Model(inputs=inputs,outputs=outputs)
    model.summary()
    model.compile(optimizer=keras.optimizers.Adam(lr=0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-8),
              loss='categorical_crossentropy',
              metrics=['acc'])
    #二、模型训练及指标返回
    history = model.fit(train_data,train_label,batch_size=batch_size,validation_data=(val_data,val_label),callbacks=callbacks,epochs=epochs,verbose=verbose)
    train_acc = history.history['acc']
    val_acc = history.history['val_acc']
    test_loss,test_acc = model.evaluate(test_data,test_label)
    return model,train_acc,val_acc,test_acc

def DCNN(train,test,val,output_categories,num_words,maxlen,embedding_dim,callbacks,
         kernels=[2,3,4],filters=128,dropout_rate=0.5,k_maxpooling=3,dense_units=256,
         batch_size=512,epochs=10,verbose=1):
    """
    TextCNN模型：DCNN，Conv1D + KMaxPooling1D
    :param train: 训练数据，格式：(data,label)
    :param test: 测试数据，格式：(data,label)
    :param val: 验证数据，格式：(data,label)
    :param output_categories: 分类类别数
    :param num_words: 限制top_n词数（文本内仅计算top_n的词向量）
    :param maxlen: 最大句长限制（计算基本单元：词组）
    :param embedding_dim: 词向量维度
    :param callbacks: 每个epoch进行回调计算的指标
    :param kernels: 卷积核数量list类型
    :param filters: 卷积层输出维度
    :param dropout_rate:
    :param k_maxpooling: 保留top_k的MaxPooling结果
    :param dense_units: Dense层神经元数量
    :param batch_size:
    :param epochs:
    :param verbose:
    :return: model：模型；train_acc：训练精度；val_acc：验证精度；test_acc：测试精度
    """
    #一、数据准备
    train_data,train_label = train
    test_data,test_label = test
    val_data,val_label = val
    #一、网络构建
    inputs = Input(shape=(maxlen,))
    embedding = Embedding(input_dim=num_words,input_length=maxlen,output_dim=embedding_dim,trainable=True)(inputs)
    pool_output = []
    for kernel_size in kernels:
        conv = Conv1D(filters=filters,kernel_size=kernel_size,strides=1,padding='same',activation='relu')(embedding)
        pooling = KMaxPooling1D(k=k_maxpooling)(conv)
        pool_output.append(pooling)
    pool_output = concatenate(pool_output)
    dropout = Dropout(dropout_rate)(pool_output)
    flatten = Flatten()(dropout)
    dense = Dense(dense_units,activation='relu')(flatten)
    outputs = Dense(output_categories,activation='softmax')(dense)
    model = Model(inputs=inputs,outputs=outputs)
    model.summary()
    model.compile(optimizer=keras.optimizers.Adam(lr=0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-8),
              loss='categorical_crossentropy',
              metrics=['acc'])
    #二、模型训练及指标返回
    history = model.fit(train_data,train_label,batch_size=batch_size,validation_data=(val_data,val_label),callbacks=callbacks,epochs=epochs,verbose=verbose)
    train_acc = history.history['acc']
    val_acc = history.history['val_acc']
    test_loss,test_acc = model.evaluate(test_data,test_label)
    return model,train_acc,val_acc,test_acc

def CGRUSeries(train,test,val,output_categories,num_words,maxlen,embedding_dim,callbacks,
               filters=128,kernel_size=3,gru_units=256,dropout=0.2,recurrent_dropout=0.2,
               batch_size=512,epochs=10,verbose=1):
    """
    C-LSTM模型：C-GRU串联
    :param train: 训练数据，格式：(data,label)
    :param test: 测试数据，格式：(data,label)
    :param val: 验证数据，格式：(data,label)
    :param output_categories: 分类类别数
    :param num_words: 限制top_n词数（文本内仅计算top_n的词向量）
    :param maxlen: 最大句长限制（计算基本单元：词组）
    :param embedding_dim: 词向量维度
    :param callbacks: 每个epoch进行回调计算的指标
    :param filters: 卷积层输出维度
    :param kernel_size: 卷积核数量
    :param gru_units: GRU层神经元数量
    :param dropout:
    :param recurrent_dropout:
    :param batch_size:
    :param epochs:
    :param verbose:
    :return: model：模型；train_acc：训练精度；val_acc：验证精度；test_acc：测试精度
    """
    #一、数据准备
    train_data,train_label = train
    test_data,test_label = test
    val_data,val_label = val
    #一、网络构建
    inputs = Input(shape=(maxlen,))
    embedding = Embedding(input_dim=num_words,input_length=maxlen,output_dim=embedding_dim,trainable=True)(inputs)
    conv = Conv1D(filters=filters,kernel_size=kernel_size,strides=1,padding='same',activation='relu')(embedding)
    pooling = MaxPooling1D()(conv)
    bigru = Bidirectional(GRU(units=gru_units,dropout=dropout,recurrent_dropout=recurrent_dropout,return_sequences=True))(pooling)
    bigru = Bidirectional(GRU(units=gru_units,dropout=dropout,recurrent_dropout=recurrent_dropout,activation='relu'))(bigru)
    outputs = Dense(output_categories,activation='softmax')(bigru)
    model = Model(inputs=inputs,outputs=outputs)
    model.summary()
    model.compile(optimizer=keras.optimizers.Adam(lr=0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-8),
              loss='categorical_crossentropy',
              metrics=['acc'])
    #二、模型训练及指标返回
    history = model.fit(train_data,train_label,batch_size=batch_size,validation_data=(val_data,val_label),callbacks=callbacks,epochs=epochs,verbose=verbose)
    train_acc = history.history['acc']
    val_acc = history.history['val_acc']
    test_loss,test_acc = model.evaluate(test_data,test_label)
    return model,train_acc,val_acc,test_acc

def CGRUParallel(train,test,val,output_categories,num_words,maxlen,embedding_dim,callbacks,
                 filters=128,kernel_size=3,gru_units=256,dropout=0.2,recurrent_dropout=0.2,dense_units=256,
                 batch_size=512,epochs=10,verbose=1):
    """
    C-LSTM模型：C-GRU并联
    :param train: 训练数据，格式：(data,label)
    :param test: 测试数据，格式：(data,label)
    :param val: 验证数据，格式：(data,label)
    :param output_categories: 分类类别数
    :param num_words: 限制top_n词数（文本内仅计算top_n的词向量）
    :param maxlen: 最大句长限制（计算基本单元：词组）
    :param embedding_dim: 词向量维度
    :param callbacks: 每个epoch进行回调计算的指标
    :param filters: 卷积层输出维度
    :param kernel_size: 卷积核数量
    :param gru_units: GRU层神经元数量
    :param dropout:
    :param recurrent_dropout:
    :param dense_units: Dense层神经元数量
    :param batch_size:
    :param epochs:
    :param verbose:
    :return: model：模型；train_acc：训练精度；val_acc：验证精度；test_acc：测试精度
    """
    #一、数据准备
    train_data,train_label = train
    test_data,test_label = test
    val_data,val_label = val
    #一、网络构建
    inputs = Input(shape=(maxlen,))
    embedding = Embedding(input_dim=num_words,input_length=maxlen,output_dim=embedding_dim,trainable=True)(inputs)
    conv = Conv1D(filters=filters,kernel_size=kernel_size,strides=1,padding='same',activation='relu')(embedding)
    pooling = MaxPooling1D()(conv)
    flatten = Flatten()(pooling)
    cnn_dense = Dense(dense_units,activation='relu')(flatten)
    bigru = Bidirectional(GRU(units=gru_units,dropout=dropout,recurrent_dropout=recurrent_dropout,return_sequences=True))(embedding)
    bigru = Bidirectional(GRU(units=gru_units,dropout=dropout,recurrent_dropout=recurrent_dropout,activation='relu'))(bigru)
    rnn_dense = Dense(dense_units,activation='relu')(bigru)
    concate = concatenate([cnn_dense,rnn_dense])
    outputs = Dense(output_categories,activation='softmax')(concate)
    model = Model(inputs=inputs,outputs=outputs)
    model.summary()
    model.compile(optimizer=keras.optimizers.Adam(lr=0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-8),
              loss='categorical_crossentropy',
              metrics=['acc'])
    #二、模型训练及指标返回
    history = model.fit(train_data,train_label,batch_size=batch_size,validation_data=(val_data,val_label),callbacks=callbacks,epochs=epochs,verbose=verbose)
    train_acc = history.history['acc']
    val_acc = history.history['val_acc']
    test_loss,test_acc = model.evaluate(test_data,test_label)
    return model,train_acc,val_acc,test_acc

def plot_accuracy(models):
    """
    绘制模型结果：模型数量不能超过len(colors)
    :param models: 模型组list类型
    :return:
    """
    colors = ['c','g','r','b','m','y','k','golden','navy','deepskyblue','coral']
    plt.rcParams['figure.figsize'] = (16,12)
    print('*-'*60+'*')
    print('Test accuracy:')
    for model in models:
        print(model,':',eval(model+'_test_acc'))
        train_acc = eval(model+'_train_acc')
        val_acc = eval(model+'_val_acc')
        #plt.plot(range(1,len(train_acc)+1),train_acc,color=colors[0],marker='o',linestyle='-',label=model+':train')
        plt.plot(range(1,len(val_acc)+1),val_acc,color=colors[0],marker='o',linestyle=':',label=model+':validation')
        colors = colors[1:]
    print('*-'*60+'*')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    return

if __name__ == '__main__':
    # 一、数据加载与处理

    # 0. 工作路径
    root_dir = '/home1/liushaoweihua/nlp_tools/project/aicai/model'
    sys.path.append(root_dir)
    os.chdir(root_dir)
    #os.environ['CUDA_VISIBLE_DEVICES'] = '0,1' #指定GPU

    # 1. 数据加载
    file_path = '../data/aicai_sample.xlsx'
    file_df = pd.read_excel(file_path,usecols=[1,2],names=['text','intent'])
    data = file_df.text.values
    label = file_df.intent.values

    data_dict = {}
    for data_i,label_i in zip(data,label):
        data_dict.setdefault(label_i,[]).append(data_i)

    sample_size = 20
    seed = 0
    test_rate = 0.1
    train,test = resample(data_dict=data_dict,sample_size=sample_size,seed=seed,test_rate=test_rate)

    # 2. 数据处理
    tqdm_train = tqdm(train)
    tqdm_test = tqdm(test)
    tqdm_train.set_description("Train:")
    tqdm_train.set_description("Test:")
    train_label = []
    train_data = []
    test_label = []
    test_data = []

    try:
        for item in tqdm_train:
            data = item[0]
            label = item[1]
            train_label.append(label)
            train_data.append(text_tokenizer([data])[0])
    except KeyboardInterrupt:
        tqdm.close()
        raise
    try:
        for item in tqdm_test:
            data = item[0]
            label = item[1]
            test_label.append(label)
            test_data.append(text_tokenizer([data])[0])
    except KeyboardInterrupt:
        tqdm.close()
        raise

    label_set = []
    for label in train_label:
        if label not in label_set:
            label_set.append(label)

    label_convert = dict([[item,label_set.index(item)] for item in label_set])
    print(label_convert)
    label_reconvert = dict([(label_convert[key],key) for key in label_convert])
    print(label_reconvert)

    train_label = [label_convert[item] for item in train_label]
    test_label = [label_convert[item] for item in test_label]
    train_label = to_categorical(train_label)
    test_label = to_categorical(test_label)

    # 3. 文本序列化
    num_words = 10000
    maxlen = 25
    embedding_dim = 128
    output_categories = len(label_convert)

    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(train_data)
    train_data = tokenizer.texts_to_sequences(train_data)
    test_data = tokenizer.texts_to_sequences(test_data)
    train_data = pad_sequences(train_data,maxlen=maxlen)
    test_data = pad_sequences(test_data,maxlen=maxlen)

    # 4. 输出指标
    target_names = [label_reconvert[i] for i in range(len(label_reconvert))]
    print(target_names)
    classication_report = ClassificationReport(target_names=target_names)

    # 二、模型训练

    # 1. 设置基本超参数（未调参）
    basic_hyper_parameters = {
        'train': (train_data, train_label),
        'test': (test_data, test_label),
        'val': (test_data, test_label),
        'output_categories': output_categories,
        'num_words': num_words,
        'maxlen': maxlen,
        'embedding_dim': embedding_dim,
        'callbacks': [classication_report]
    }
    transformer, transformer_train_acc, transformer_val_acc, transformer_test_acc = Transformer(**basic_hyper_parameters)
    bigru, bigru_train_acc, bigru_val_acc, bigru_test_acc = BiGRU(**basic_hyper_parameters)
    textcnn, textcnn_train_acc, textcnn_val_acc, textcnn_test_acc = TextCNN(**basic_hyper_parameters)
    dcnn, dcnn_train_acc, dcnn_val_acc, dcnn_test_acc = DCNN(**basic_hyper_parameters)
    cgruseries, cgruseries_train_acc, cgruseries_val_acc, cgruseries_test_acc = CGRUSeries(**basic_hyper_parameters)
    cgruparallel, cgruparallel_train_acc, cgruparallel_val_acc, cgruparallel_test_acc = CGRUParallel(**basic_hyper_parameters)

    # 2. 模型结果绘制
    models = ['transformer', 'bigru', 'textcnn', 'dcnn', 'cgruseries', 'cgruparallel']
    plot_accuracy(models=models)

    # 3. 模型预测
    news = """
    恭喜您获得月末限量放款名额！
    """
    news = text_tokenizer([news])
    news = tokenizer.texts_to_sequences(news)
    news = pad_sequences(news, maxlen=maxlen)
    for model in models:
        print(model, ':', label_reconvert[eval(model).predict(news).argmax()])