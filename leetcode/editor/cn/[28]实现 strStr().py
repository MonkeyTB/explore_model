# 实现 strStr() 函数。 
# 
#  给你两个字符串 haystack 和 needle ，请你在 haystack 字符串中找出 needle 字符串出现的第一个位置（下标从 0 开始）。如
# 果不存在，则返回 -1 。 
# 
#  
# 
#  说明： 
# 
#  当 needle 是空字符串时，我们应当返回什么值呢？这是一个在面试中很好的问题。 
# 
#  对于本题而言，当 needle 是空字符串时我们应当返回 0 。这与 C 语言的 strstr() 以及 Java 的 indexOf() 定义相符。 
# 
#  
# 
#  示例 1： 
# 
#  
# 输入：haystack = "hello", needle = "ll"
# 输出：2
#  
# 
#  示例 2： 
# 
#  
# 输入：haystack = "aaaaa", needle = "bba"
# 输出：-1
#  
# 
#  示例 3： 
# 
#  
# 输入：haystack = "", needle = ""
# 输出：0
#  
# 
#  
# 
#  提示： 
# 
#  
#  0 <= haystack.length, needle.length <= 5 * 104 
#  haystack 和 needle 仅由小写英文字符组成 
#  
#  Related Topics 双指针 字符串 字符串匹配 
#  👍 1166 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
# class Solution:
#     def strStr(self, haystack: str, needle: str) -> int:
#         '''
#         8:37	info
# 		解答成功:
# 		执行耗时:48 ms,击败了56.72% 的Python3用户
# 		内存消耗:15.1 MB,击败了63.83% 的Python3用户
#         方法：
#             字符串切片比较
#         '''
#         if len(needle) > len(haystack) : return -1
#         for i in range(0,len(haystack) - len(needle) + 1,1):
#             if haystack[i : (i+len(needle))] == needle : return i
#         return -1
class Solution:
    def next(self, s):
        a = len(s)
        next = ['' for i in range(a)]
        j, k = 0, -1
        next[0] = k
        while (j < a - 1):
            if k == -1 or s[k] == s[j]:
                k += 1
                j += 1
                next[j] = k
            else:
                k = next[k]
        return next

    def strStr(self, haystack: str, needle: str) -> int:
        '''
        9:11	info
		解答成功:
		执行耗时:84 ms,击败了5.68% 的Python3用户
		内存消耗:15.9 MB,击败了14.23% 的Python3用户
		方法：
		    KMP算法
        '''
        if len(needle) == 0: return 0
        next = self.next(needle)
        i, j = 0, 0
        while i < len(haystack) and j < len(needle):
            if haystack[i] == needle[j] or j == -1:
                i += 1
                j += 1
            else:
                j = next[j]
        if j == len(needle):
            return i-j
        else:
            return -1



# leetcode submit region end(Prohibit modification and deletion)
