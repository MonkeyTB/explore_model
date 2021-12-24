# 给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串 s ，判断字符串是否有效。 
# 
#  有效字符串需满足： 
# 
#  
#  左括号必须用相同类型的右括号闭合。 
#  左括号必须以正确的顺序闭合。 
#  
# 
#  
# 
#  示例 1： 
# 
#  
# 输入：s = "()"
# 输出：true
#  
# 
#  示例 2： 
# 
#  
# 输入：s = "()[]{}"
# 输出：true
#  
# 
#  示例 3： 
# 
#  
# 输入：s = "(]"
# 输出：false
#  
# 
#  示例 4： 
# 
#  
# 输入：s = "([)]"
# 输出：false
#  
# 
#  示例 5： 
# 
#  
# 输入：s = "{[]}"
# 输出：true 
# 
#  
# 
#  提示： 
# 
#  
#  1 <= s.length <= 104 
#  s 仅由括号 '()[]{}' 组成 
#  
#  Related Topics 栈 字符串 
#  👍 2847 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def isValid(self, s: str) -> bool:
        '''
        8:46	info
		解答成功:
		执行耗时:24 ms,击败了97.63% 的Python3用户
		内存消耗:15.2 MB,击败了5.46% 的Python3用户
        方法：
            构造字典，将添加的左符号变为右符号，方便直接判断
        '''
        res = []
        dict_ = {'(':')','[':']','{':'}'}
        for i in s:
            if i in ['(','[','{']: # 左括号，添加到res
                res.append(dict_[i])
            elif i in [')',']','}'] and len(res) > 0 and res[-1] == i: # 右括号，且res最后一个和i相同
                res.pop()
            elif len(res) == 0 and i in [')',']','}'] : # 右括号，且res为空
                return False
            elif len(res) > 0 and i != res[-1]: # res不为空，但先来一个不匹配的i
                return False
        if len(res) == 0: return True
        else: return False


# leetcode submit region end(Prohibit modification and deletion)
