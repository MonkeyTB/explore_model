# 给定一个字符串 s 和一个整数 k，从字符串开头算起，每计数至 2k 个字符，就反转这 2k 字符中的前 k 个字符。 
# 
#  
#  如果剩余字符少于 k 个，则将剩余字符全部反转。 
#  如果剩余字符小于 2k 但大于或等于 k 个，则反转前 k 个字符，其余字符保持原样。 
#  
# 
#  
# 
#  示例 1： 
# 
#  
# 输入：s = "abcdefg", k = 2
# 输出："bacdfeg"
#  
# 
#  示例 2： 
# 
#  
# 输入：s = "abcd", k = 2
# 输出："bacd"
#  
# 
#  
# 
#  提示： 
# 
#  
#  1 <= s.length <= 104 
#  s 仅由小写英文组成 
#  1 <= k <= 104 
#  
#  Related Topics 双指针 字符串 
#  👍 218 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    '''
    8:41	info
		解答成功:
		执行耗时:32 ms,击败了80.11% 的Python3用户
		内存消耗:15.1 MB,击败了77.92% 的Python3用户
		注意：
		再循环上下手，而不要去计算位置再位置内计算，直接再for训练里面每次跳2k即可，另外需要注意，转换为列表做
    '''
    def reverseString(self, s):
        left, right = 0, len(s) - 1
        while left < right:
            s[left], s[right] = s[right], s[left]
            left += 1
            right -= 1
        return s
    def reverseStr(self, s: str, k: int) -> str:
        res = list(s)
        for i in range(0, len(s), 2*k):
            res[i:i+k] = self.reverseString(res[i:i+k])
        return ''.join(res)
# leetcode submit region end(Prohibit modification and deletion)
