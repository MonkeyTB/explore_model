# LSTM+CRF 优化
+ 从输入添加了词向量，词向量为垂直领域训练，效果提升巨大，尤其在短文本的NER识别中（提升20-30个点），长文本NER识别中取得了优于四层BERT+Global Pointer的结果
# 文件
+ /data/wv/ 中存储训练好的词向量模型

# 指标展示
![img.png](img.png)