#
# @lc app=leetcode.cn id=16 lang=python3
#
# [16] 最接近的三数之和
#

# @lc code=start
from typing import List
import math
class Solution:
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        # 三数之和的改编，只需要记录一个最小值就行abs(target-sum)
        nums.sort()
        total = math.inf # 记录最接近的值
        for i in range(len(nums)):
            if i > 0 and nums[i] == nums[i-1]:
                continue
            left, right = i+1, len(nums)-1
            while left < right:
                mid = nums[i] + nums[left] + nums[right]
                if abs(mid-target) < abs(total-target):
                    total = mid
                elif mid > target:
                    right -= 1
                else:
                    left += 1
        return total



# @lc code=end
s = Solution()
s.threeSumClosest([4,0,5,-5,3,3,0,-4,-5], -2)
