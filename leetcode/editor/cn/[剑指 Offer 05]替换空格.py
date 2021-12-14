# 请实现一个函数，把字符串 s 中的每个空格替换成"%20"。 
# 
#  
# 
#  示例 1： 
# 
#  输入：s = "We are happy."
# 输出："We%20are%20happy." 
# 
#  
# 
#  限制： 
# 
#  0 <= s 的长度 <= 10000 
#  Related Topics 字符串 
#  👍 189 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def replaceSpace(self, s: str) -> str:
        '''
        8:48	info
		解答成功:
		执行耗时:24 ms,击败了96.70% 的Python3用户
		内存消耗:14.9 MB,击败了74.73% 的Python3用户
        方法：
            统计空格个数
            双指针移动，操作list
        '''
        num = s.count(' ')
        res = list(s)
        res.extend([' '] * num * 2)

        left, right = len(s) - 1, len(res) - 1
        while left >= 0:
            if res[left] == ' ':
                res[right-2:right+1] = '%20'
                right -= 3
            else:
                res[right] = res[left]
                right -= 1
            left -= 1
        return ''.join(res)

# leetcode submit region end(Prohibit modification and deletion)
