# 给定一个非空的字符串，判断它是否可以由它的一个子串重复多次构成。给定的字符串只含有小写英文字母，并且长度不超过10000。 
# 
#  示例 1: 
# 
#  
# 输入: "abab"
# 
# 输出: True
# 
# 解释: 可由子字符串 "ab" 重复两次构成。
#  
# 
#  示例 2: 
# 
#  
# 输入: "aba"
# 
# 输出: False
#  
# 
#  示例 3: 
# 
#  
# 输入: "abcabcabcabc"
# 
# 输出: True
# 
# 解释: 可由子字符串 "abc" 重复四次构成。 (或者子字符串 "abcabc" 重复两次构成。)
#  
#  Related Topics 字符串 字符串匹配 
#  👍 580 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def get_next(self, sub):
        next = [0]*len(sub)
        j = 0
        for i in range(1, len(sub), 1):
            while sub[i] != sub[j] and j > 0:
                j = next[j-1]
            if sub[i] == sub[j]:
                j += 1
            next[i] = j
        return next
    def repeatedSubstringPattern(self, s: str) -> bool:
        '''
        22:41	info
			解答成功:
			执行耗时:136 ms,击败了32.90% 的Python3用户
			内存消耗:15.3 MB,击败了8.44% 的Python3用户
        方法：
            KMP算法，next数组记录了重复情况，如果next[-1] ！= 0 并且字符串长度和最大重复字串能够整除
        '''
        if len(s) == 0: return False
        next = self.get_next(s)
        if (next[-1] != 0 and len(s) % (len(s) - next[-1]) == 0):
            return True
        return False

# leetcode submit region end(Prohibit modification and deletion)
