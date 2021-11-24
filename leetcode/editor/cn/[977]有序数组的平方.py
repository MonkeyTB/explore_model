# 给你一个按 非递减顺序 排序的整数数组 nums，返回 每个数字的平方 组成的新数组，要求也按 非递减顺序 排序。 
# 
#  
#  
# 
#  
# 
#  示例 1： 
# 
#  
# 输入：nums = [-4,-1,0,3,10]
# 输出：[0,1,9,16,100]
# 解释：平方后，数组变为 [16,1,0,9,100]
# 排序后，数组变为 [0,1,9,16,100] 
# 
#  示例 2： 
# 
#  
# 输入：nums = [-7,-3,2,3,11]
# 输出：[4,9,9,49,121]
#  
# 
#  
# 
#  提示： 
# 
#  
#  1 <= nums.length <= 104 
#  -104 <= nums[i] <= 104 
#  nums 已按 非递减顺序 排序 
#  
# 
#  
# 
#  进阶： 
# 
#  
#  请你设计时间复杂度为 O(n) 的算法解决本问题 
#  
#  Related Topics 数组 双指针 排序 
#  👍 350 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def sortedSquares(self, nums: List[int]) -> List[int]:
        '''
        23:50	info
			解答成功:
			执行耗时:68 ms,击败了51.94% 的Python3用户
			内存消耗:16.4 MB,击败了33.31% 的Python3用户
        '''
        nums = [i*i for i in nums]
        res = [-1]*len(nums)
        i, j = 0, len(nums) - 1
        index = len(nums) - 1
        while i <= j:
            if nums[j] >= nums[i]:
                res[index] = nums[j]
                index -= 1
                j -= 1
            else:
                res[index] = nums[i]
                index -= 1
                i += 1
        return res
# leetcode submit region end(Prohibit modification and deletion)
