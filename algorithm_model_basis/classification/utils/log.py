# _*_coding:utf-8_*_
# 作者     ：YiSan
# 创建时间  ：2021/4/8 16:51 
# 文件     ：log.py
# IDE     : PyCharm

from datetime import datetime
from utils.config import *
def log_msg(file_ = None, kind_=None, info_=None):
    """
    日志输出模版函数
    :param file_: 日志文件写入路径
    :param kind_: 错误类型
    :param info_: 错误信息
    :return:
    """
    current = str(datetime.now())[:23].replace('.', ':')
    info_str = ' {curr_time} INFO :[{kind_}]  {info_}\n'.format(
        curr_time=current, kind_=kind_, info_=info_)
    with open(file_, encoding='utf8', mode='a') as f:
        f.writelines(info_str)
def log(str):
    log_msg(file_=config.log_path, info_ = str)