# _*_coding:utf-8_*_
# 作者     ：
# 创建时间  ：2021/5/21 14:17
# 文件     ：eval.py
# IDE     : PyCharm
import os


def conlleval(label_predict, label_path, metric_path):
    """

    :param label_predict:
    :param label_path:
    :param metric_path:
    :return:
    """
    eval_perl = "./conlleval.pl"
    with open(label_path, "w", encoding='utf8') as fw:
        line = []
        for sent_result in label_predict:
            for char, tag, tag_ in [sent_result]:
                tag = '0' if tag == 'O' else tag
#                 char = char.encode("utf-8")
                line.append("{}\t{}\t{}\n".format(char, tag, tag_))
            line.append("\n")
        fw.writelines(line)
    os.system("perl {} < {} > {}".format(eval_perl, label_path, metric_path))
    with open(metric_path) as fr:
        metrics = [line.strip() for line in fr]
    return metrics

if __name__ == '__main__':
    model_predict = [['北', 'B-localtion','B-localtion'],['京', 'I-localtion','B-localtion'],['市', 'E-localtion','E-localtion'],['塑', 'B-kernel','B-kernel'],['胶', 'E-kernel','E-kernel']]
    conlleval(model_predict,'1.txt','2.txt')