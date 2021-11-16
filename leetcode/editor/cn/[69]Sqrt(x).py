# 给你一个非负整数 x ，计算并返回 x 的 算术平方根 。 
# 
#  由于返回类型是整数，结果只保留 整数部分 ，小数部分将被 舍去 。 
# 
#  注意：不允许使用任何内置指数函数和算符，例如 pow(x, 0.5) 或者 x ** 0.5 。 
# 
#  
# 
#  示例 1： 
# 
#  
# 输入：x = 4
# 输出：2
#  
# 
#  示例 2： 
# 
#  
# 输入：x = 8
# 输出：2
# 解释：8 的算术平方根是 2.82842..., 由于返回类型是整数，小数部分将被舍去。
#  
# 
#  
# 
#  提示： 
# 
#  
#  0 <= x <= 231 - 1 
#  
#  Related Topics 数学 二分查找 
#  👍 825 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def mySqrt(self, x: int) -> int:
        '''
        23:00	info
			解答成功:
			执行耗时:32 ms,击败了89.74% 的Python3用户
			内存消耗:15 MB,击败了25.00% 的Python3用户
        二分查找
        0，1 单独处理
        题目要求返回值去掉小数点之后的，因此当left 和 right 中间的值才能满足值的时候，返回left即可
        x = 5   sqrt(x) = 2.236   left 指向2  ， right 指向3
        '''
        if x == 0 : return 0
        if x == 1 : return 1
        left, right = -1, x
        while left + 1 < right:
            mid = (left + right + 1) // 2
            if mid * mid == x:
                return mid
            elif mid * mid > x:
                right = mid
            else:
                left = mid
        return left
# leetcode submit region end(Prohibit modification and deletion)
