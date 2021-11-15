# _*_coding:utf-8_*_
# 作者     ：YiSan
# 创建时间  ：2021/11/3 10:02 
# 文件     ：eval.py
# IDE     : PyCharm
import pandas as pd
from sklearn.metrics import classification_report

path = r'优化测试样本分析-数据返回2021.11.2.xlsx'
df = pd.read_excel(io=path,sheet_name='2000')
mark = df['标注结果'].tolist()
pred = df['label'].tolist()
dict_target = {'职位名称':0,'要求':1,'描述':2,'其他口号':3,'薪资福利':4}
target_name = ['职位名称','要求','描述','其他口号','薪资福利']
marks, preds = [], []
for i in mark:
    marks.append(dict_target[i])
for j in pred:
    preds.append(dict_target[j])

print( classification_report(marks,preds,target_names=target_name) )

