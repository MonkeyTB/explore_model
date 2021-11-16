# 给你一个字符串 s，找到 s 中最长的回文子串。 
# 
#  
# 
#  示例 1： 
# 
#  
# 输入：s = "babad"
# 输出："bab"
# 解释："aba" 同样是符合题意的答案。
#  
# 
#  示例 2： 
# 
#  
# 输入：s = "cbbd"
# 输出："bb"
#  
# 
#  示例 3： 
# 
#  
# 输入：s = "a"
# 输出："a"
#  
# 
#  示例 4： 
# 
#  
# 输入：s = "ac"
# 输出："a"
#  
# 
#  
# 
#  提示： 
# 
#  
#  1 <= s.length <= 1000 
#  s 仅由数字和英文字母（大写和/或小写）组成 
#  
#  Related Topics 字符串 动态规划 👍 4310 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def longestPalindrome(self, s: str) -> str:
        '''
        			解答成功:
			执行耗时:6752 ms,击败了30.72% 的Python3用户
			内存消耗:23.4 MB,击败了7.76% 的Python3用户

        动态规划
        '''
        l = len(s)
        if len(s) <= 1: return s
        f = [[False for _ in range(l)] for _ in range(l)]
        for i in range(l):
            for j in range(l):
                if i == j:
                    f[i][j] = True
        begin_position = 0
        max_length = 1
        for j in range(1,l):
            for i in range(j):
                if s[i] == s[j] and (f[i+1][j-1] or j - i <= 2): # 注意边界条件
                    f[i][j] = True
                    if j - i + 1 > max_length :
                        max_length = j - i + 1
                        begin_position = i
        return s[begin_position : begin_position+max_length]


# leetcode submit region end(Prohibit modification and deletion)
ob= Solution()
print( ob.longestPalindrome('cbbd') )