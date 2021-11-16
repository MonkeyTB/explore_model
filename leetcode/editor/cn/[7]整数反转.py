# 给你一个 32 位的有符号整数 x ，返回将 x 中的数字部分反转后的结果。 
# 
#  如果反转后整数超过 32 位的有符号整数的范围 [−2³¹, 231 − 1] ，就返回 0。 
# 假设环境不允许存储 64 位整数（有符号或无符号）。
# 
#  
# 
#  示例 1： 
# 
#  
# 输入：x = 123
# 输出：321
#  
# 
#  示例 2： 
# 
#  
# 输入：x = -123
# 输出：-321
#  
# 
#  示例 3： 
# 
#  
# 输入：x = 120
# 输出：21
#  
# 
#  示例 4： 
# 
#  
# 输入：x = 0
# 输出：0
#  
# 
#  
# 
#  提示： 
# 
#  
#  -2³¹ <= x <= 2³¹ - 1 
#  
#  Related Topics 数学 👍 3226 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
import math
class Solution:
    def reverse(self, x: int) -> int:
        '''
        			解答成功:
			执行耗时:40 ms,击败了26.36% 的Python3用户
			内存消耗:14.7 MB,击败了96.99% 的Python3用户

		1. 记录标注位
		2. 计算反转
        '''
        flag = 1 if x > 0 else -1
        x = abs(x)
        res = 0
        while(x != 0):
            res = res*10 + x%10
            x = x//10
        if -math.pow(2,31) <= res*flag <= math.pow(2,31)-1:
            return res*flag
        return 0
# leetcode submit region end(Prohibit modification and deletion)
ob = Solution()
print( ob.reverse(-123) )