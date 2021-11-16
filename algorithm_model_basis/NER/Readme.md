## 20210616_01_YiSan
### 环境配置
```buildoutcfg
tensorflow                2.1.0                    pypi_0    pypi
tensorflow-addons         0.9.1                    pypi_0    pypi
```
### 使用说明
#### 参数配置
+ args_help.py
    + 可配置参数
        + output_dir     ： checkpoint 模型文件保存地址
        + savemode_dir   ： pb 模型文件保存地址
        + PbFlag         ： True 保存 pd 模型，False 保存 checkpoints 模型，注意，True的时候需要再model.py打开call函数前面的@tf.function
        + epoch          ： 迭代次数
        + lr             ： 学习率
        + embedding_size ：嵌入向量的维度
+ 数据
    + 必要文件
        + train.txt  :  训练集数据
        + test.txt   :  测试集数据
    + 数据格式
        ```text
        凤 B-localtion
        台 I-localtion
        县 E-localtion
        丛 B-kernel
        绿 E-kernel
        果 B-industry
        蔬 I-industry
        种 I-industry
        植 E-industry
        专 B-organization
        业 I-organization
        合 I-organization
        作 I-organization
        社 E-organization
        end
        大 B-localtion
        连 E-localtion
        新 B-kernel
        乐 E-kernel
        置 B-industry
        业 I-industry
        经 I-industry
        纪 E-industry
        有 B-organization
        限 I-organization
        公 I-organization
        司 E-organization
        end
        ```
    + 最终会生成两个文件
        + tags.txt  ： 标签
            ```text
            B-industry
            I-industry
            E-industry
            S-industry
            ...
            ```
        + vocab.txt ： 词表
            ```text
            费
            焉
            积
            盈
            用
            熙
            目
            ...
            ```
#### 训练
+ train.py
    + python train.py，会根据参数训练模型并进行保存，日志文件写进 log/ 文件
#### 测试
+ test.py 
    + python test.py，批量测试结果，默认加载 checkpoints 模型文件，结果写在 result/text.txt 中。
#### 预测
+ predict.py
    + 单挑预测，支持 checkpoints 模型和 pb 模型，结果如下：
    ```json
    文成县南田镇经济开发总公司乡镇企业服务部
    [{
        "begin": 1,
        "end": 6,
        "words": "文成县南田镇",
        "type": "localtion"
    },
    {
        "begin": 7,
        "end": 10,
        "words": "经济开发",
        "type": "industry"
    },
    {
        "begin": 11,
        "end": 13,
        "words": "总公司",
        "type": "organization"
    },
    {
        "begin": 14,
        "end": 15,
        "words": "乡镇",
        "type": "2_localtion"
    },
    {
        "begin": 16,
        "end": 20,
        "words": "企业服务部",
        "type": "2_organization"
    }]
    ```
