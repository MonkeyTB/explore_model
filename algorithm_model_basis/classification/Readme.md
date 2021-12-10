# 20210928_01_YISAN
### 环境配置

```buildoutcfg
tensorflow                2.1.0                    pypi_0    pypi
tensorflow-addons         0.9.1                    pypi_0    pypi
```

### 使用说明             

#### 训练
```python
python main.py
```
#### 单挑/批量测试
```python
python online_test.py
```
#### 部署模型平台测试
```python
python tf-server.py
```
#### config.py
+ model_type:可选模型配置参数
    + LR : 逻辑斯蒂回归
    + XGBoost : GBDT
    + TextCnn : text cnn model
    + TextCnnNew : text cnn multi【自研模型，效果待测试，目前没有发现超过text cnn的数据和场景】
    + TextCnnMultiDim ： text cnn  字、词、词性向量 三种编码方式一起预测，在短文本预测中效果更好，长文本根据实际情况验证
    + RCNN : text RCNN model
    + TextRNN : text rnn model
    + TextAttBiRNN : lstm attention model
+ stop_word : 停用词标志位
    + True : 加载停用词
    + False ：不加载停用词
+ id_type ： embedding类型
    + tf-idf ： tf-idf编码方式【LR、XGBoost】
    + word-count ： 词频编码方式【LR、XGBoost】
    + word2vec ： word2vec编码方式【TextCnn、TextCnnNew】
+ numclass ： 分类类别数
+ embedding_size ： 向量维度
+ dict_lable ：标签为汉字对应的字典
+ ch_label_type : 标签是否为汉字
    + True ： 汉字
    + False : 数字

# 20211210_02_YiSan
+ 添加修正的交叉熵損失在model.py中，各個模型可以靈活替換，參考來源![鏈接](https://spaces.ac.cn/archives/4293)
