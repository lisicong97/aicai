###  模型内容包括：

#### 文件夹

**model：**main函数和各自定义的Keras-Layer

**data：**包含data文件

#### 主体部分main.py

模型是用Keras写的，分别为Transformer, BiGRU, TextCNN, DCNN, CGRUSeries, CGRUParallel

**main函数：**

工作路径：root_dir = '/home1/liushaoweihua/nlp_tools/project/text_classification/model'修改为模型所在路径；

file_path = '../data/aicai_sample/aicai_sample.xlsx'修改为数据所在路径；

基础超参数设置：

```python
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
```

目前仅测试了基础超参数，未进行调参。

#### 终端下报错

原因：终端下无法进行matplotlib绘图。