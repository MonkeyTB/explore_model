# 字符串的左旋转操作是把字符串前面的若干个字符转移到字符串的尾部。请定义一个函数实现字符串左旋转操作的功能。比如，输入字符串"abcdefg"和数字2，该函数
# 将返回左旋转两位得到的结果"cdefgab"。 
# 
#  
# 
#  示例 1： 
# 
#  输入: s = "abcdefg", k = 2
# 输出: "cdefgab"
#  
# 
#  示例 2： 
# 
#  输入: s = "lrloseumgh", k = 6
# 输出: "umghlrlose"
#  
# 
#  
# 
#  限制： 
# 
#  
#  1 <= k < s.length <= 10000 
#  
#  Related Topics 数学 双指针 字符串 
#  👍 175 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def reverseLeftWords_1(self, s: str, n: int) -> str:
        '''
        8:38	info
		解答成功:
		执行耗时:32 ms,击败了74.07% 的Python3用户
		内存消耗:15.1 MB,击败了17.70% 的Python3用户
        注意：
            搞笑解法
        '''
        return s[n:len(s)] + s[0:n]
    def reverseLeftWords(self, s: str, n: int) -> str:
        '''
        8:53	info
		解答成功:
		执行耗时:40 ms,击败了29.10% 的Python3用户
		内存消耗:15.2 MB,击败了5.09% 的Python3用户
        方法：
            通过三次反转，得到最终左旋字符串
        '''
        def reverse(s_sub, left, right):
            while left < right:
                s_sub[left], s_sub[right] = s_sub[right], s_sub[left]
                left += 1
                right -= 1
        res = list(s)
        reverse(res, 0, n - 1)
        reverse(res, n, len(s) - 1)
        reverse(res, 0, len(s) - 1)
        return ''.join( res )
# leetcode submit region end(Prohibit modification and deletion)
