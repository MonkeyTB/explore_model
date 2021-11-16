# _*_coding:utf-8_*_
# 作者     ：YiSan
# 创建时间  ：2021/5/21 14:17
# 文件     ：args_help.py
# IDE     : PyCharm
import argparse

parser = argparse.ArgumentParser(description="train")
parser.add_argument("--train_path", type=str, default="./data/train.txt",help="train file")
parser.add_argument("--test_path", type=str, default="./data/test.txt",help="test file")
parser.add_argument("--output_dir", type=str, default="checkpoints/",help="output_dir")
parser.add_argument("--PbFlag",type=bool,default=False,help="Savemodel or no")
parser.add_argument('--savemode_dir',type=str,default='savemode/',help='svaemode_dir')
parser.add_argument("--vocab_file", type=str, default="./data/vocab.txt",help="vocab_file")
parser.add_argument("--tag_file", type=str, default="./data/tags.txt",help="tag_file")
parser.add_argument("--batch_size", type=int, default=256,help="batch_size")
parser.add_argument("--hidden_num", type=int, default=512,help="hidden_num")
parser.add_argument("--embedding_size", type=int, default=300,help="embedding_size")
parser.add_argument("--embedding_file", type=str, default=None,help="embedding_file")
parser.add_argument("--epoch", type=int, default=100,help="epoch")
parser.add_argument("--lr", type=float, default=1e-3,help="lr")
parser.add_argument("--require_improvement", type=int, default=100,help="require_improvement")
parser.add_argument("--log_path",type=str,default='./log/',help='log file')
args = parser.parse_args()
