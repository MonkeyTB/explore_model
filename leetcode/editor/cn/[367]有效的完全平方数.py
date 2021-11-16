# 给定一个 正整数 num ，编写一个函数，如果 num 是一个完全平方数，则返回 true ，否则返回 false 。 
# 
#  进阶：不要 使用任何内置的库函数，如 sqrt 。 
# 
#  
# 
#  示例 1： 
# 
#  
# 输入：num = 16
# 输出：true
#  
# 
#  示例 2： 
# 
#  
# 输入：num = 14
# 输出：false
#  
# 
#  
# 
#  提示： 
# 
#  
#  1 <= num <= 2^31 - 1 
#  
#  Related Topics 数学 二分查找 
#  👍 319 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def isPerfectSquare(self, num: int)->bool:
        '''
        23:09	info
			解答成功:
			执行耗时:28 ms,击败了90.09% 的Python3用户
			内存消耗:14.9 MB,击败了68.08% 的Python3用户
        二分法 over
        '''
        if num == 0 or num == 1 : return True
        left, right = -1, num
        while left + 1 < right :
            mid = (left + right + 1) // 2
            if mid * mid == num : return True
            elif mid * mid > num : right = mid
            else: left = mid
        return False
# leetcode submit region end(Prohibit modification and deletion)
