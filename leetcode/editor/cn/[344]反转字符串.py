# 编写一个函数，其作用是将输入的字符串反转过来。输入字符串以字符数组 s 的形式给出。 
# 
#  不要给另外的数组分配额外的空间，你必须原地修改输入数组、使用 O(1) 的额外空间解决这一问题。 
# 
#  
# 
#  示例 1： 
# 
#  
# 输入：s = ["h","e","l","l","o"]
# 输出：["o","l","l","e","h"]
#  
# 
#  示例 2： 
# 
#  
# 输入：s = ["H","a","n","n","a","h"]
# 输出：["h","a","n","n","a","H"] 
# 
#  
# 
#  提示： 
# 
#  
#  1 <= s.length <= 105 
#  s[i] 都是 ASCII 码表中的可打印字符 
#  
#  Related Topics 递归 双指针 字符串 
#  👍 496 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def reverseString(self, s: List[str]) -> None:
        '''
        8:21	info
		解答成功:
		执行耗时:36 ms,击败了86.44% 的Python3用户
		内存消耗:19.3 MB,击败了24.69% 的Python3用户
        方法：
        双指针法
        '''
        """
        Do not return anything, modify s in-place instead.
        """
        left, right = 0, len(s)-1
        while left < right:
            s[left], s[right] = s[right], s[left]
            left += 1
            right -= 1
# leetcode submit region end(Prohibit modification and deletion)
