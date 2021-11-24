# 给你一个正整数 n ，生成一个包含 1 到 n2 所有元素，且元素按顺时针顺序螺旋排列的 n x n 正方形矩阵 matrix 。 
# 
#  
# 
#  示例 1： 
# 
#  
# 输入：n = 3
# 输出：[[1,2,3],[8,9,4],[7,6,5]]
#  
# 
#  示例 2： 
# 
#  
# 输入：n = 1
# 输出：[[1]]
#  
# 
#  
# 
#  提示： 
# 
#  
#  1 <= n <= 20 
#  
#  Related Topics 数组 矩阵 模拟 
#  👍 527 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def generateMatrix(self, n: int) -> List[List[int]]:
        '''
        23:09	info
			解答成功:
			执行耗时:32 ms,击败了64.52% 的Python3用户
			内存消耗:15 MB,击败了71.12% 的Python3用户
        转圈圈，左闭右开，模式不变
        '''
        res = [[0] * n for _ in range(n)]
        count = 1
        left, right, up, down = 0, n-1, 0, n-1
        while left < right and up < down:
            # 上，从左到右填充
            for i in range(left, right):
                res[up][i] = count
                count += 1
            # 右， 从上到下
            for i in range(up, down):
                res[i][right] = count
                count += 1
            # 下, 从右到左
            for i in range(right,left,-1):
                res[down][i] = count
                count += 1
            # 左, 从下到上
            for i in range(down,up,-1):
                res[i][left] = count
                count += 1
            left += 1
            right -= 1
            up += 1
            down -= 1
        if n % 2 == 1: res[n//2][n//2] = count
        return res



# leetcode submit region end(Prohibit modification and deletion)
