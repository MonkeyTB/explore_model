# 给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。 
# 
#  请必须使用时间复杂度为 O(log n) 的算法。 
# 
#  
# 
#  示例 1:
# 
#  
# 输入: nums = [1,3,5,6], target = 5
# 输出: 2
#  
# 
#  示例 2: 
# 
#  
# 输入: nums = [1,3,5,6], target = 2
# 输出: 1
#  
# 
#  示例 3: 
# 
#  
# 输入: nums = [1,3,5,6], target = 7
# 输出: 4
#  
# 
#  示例 4: 
# 
#  
# 输入: nums = [1,3,5,6], target = 0
# 输出: 0
#  
# 
#  示例 5: 
# 
#  
# 输入: nums = [1], target = 0
# 输出: 0
#  
# 
#  
# 
#  提示: 
# 
#  
#  1 <= nums.length <= 104 
#  -104 <= nums[i] <= 104 
#  nums 为无重复元素的升序排列数组 
#  -104 <= target <= 104 
#  
#  Related Topics 数组 二分查找 
#  👍 1173 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def searchInsert(self, nums, target ):
        '''

        22:15	info
			解答成功:
			执行耗时:32 ms,击败了66.40% 的Python3用户
			内存消耗:15.4 MB,击败了23.63% 的Python3用
        二分法：
        找到返回id，找不到left,right 指针再该插入位置的前后，因为我们的left和right指针从-1，len(nums)开始的，
        即可以理解为左开右开
        '''
        if target < nums[0]: return 0
        if target > nums[-1]: return len(nums)
        left, right = -1, len(nums)
        while left + 1 < right:
            L, R = False, False
            mid = (left + right + 1) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                L = True
                left = mid
            else:
                R = True
                right = mid
        return left + 1
# leetcode submit region end(Prohibit modification and deletion)
ob = Solution
print(ob.searchInsert(None,[1,3],1))