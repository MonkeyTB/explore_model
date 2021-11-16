# 给定一个字符串 s ，请你找出其中不含有重复字符的 最长子串 的长度。 
# 
#  
# 
#  示例 1: 
# 
#  
# 输入: s = "abcabcbb"
# 输出: 3 
# 解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
#  
# 
#  示例 2: 
# 
#  
# 输入: s = "bbbbb"
# 输出: 1
# 解释: 因为无重复字符的最长子串是 "b"，所以其长度为 1。
#  
# 
#  示例 3: 
# 
#  
# 输入: s = "pwwkew"
# 输出: 3
# 解释: 因为无重复字符的最长子串是 "wke"，所以其长度为 3。
#      请注意，你的答案必须是 子串 的长度，"pwke" 是一个子序列，不是子串。
#  
# 
#  示例 4: 
# 
#  
# 输入: s = ""
# 输出: 0
#  
# 
#  
# 
#  提示： 
# 
#  
#  0 <= s.length <= 5 * 10⁴ 
#  s 由英文字母、数字、符号和空格组成 
#  
#  Related Topics 哈希表 字符串 滑动窗口 👍 6390 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        '''
                			解答成功:
			执行耗时:56 ms,击败了76.68% 的Python3用户
			内存消耗:15 MB,击败了62.54% 的Python3用户

        双指针
        尾指针新加的元素包含两种情况：
        1、包含在i-j之间的元素
            1.记录当前num，和max相比
            2.移动i（头指针），直到新加的元素不包含在i-j之间
        2、不包含i-j之间的元素
            继续移动尾指针
        '''
        if s == '': return 0
        i, j = 0, 1
        max = 0
        length = len(s)
        while j < length :
            if s[j] in s[i:j]:
                max = j-i if j-i > max else max
                while(s[j] in s[i:j] and i <= j):
                    i += 1
            else:
                j += 1
        return j-i if j-i > max else max
# leetcode submit region end(Prohibit modification and deletion)
