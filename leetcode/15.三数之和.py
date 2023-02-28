#
# @lc app=leetcode.cn id=15 lang=python3
#
# [15] 三数之和
#

# @lc code=start
from typing import List
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        # 排序+双指针
        nums.sort()
        path = []
        for i in range(len(nums)):
            # 去重逻辑[-1,-1,2]
            if i > 0 and nums[i] == nums[i-1]:
                continue
            left = i + 1
            right = len(nums) - 1
            while left < right:
                if -nums[i] == nums[left] + nums[right]:
                    path.append([nums[i], nums[left], nums[right]])
                    # 去重逻辑
                    while nums[left] == nums[left+1] and left + 1 < right:
                        left += 1
                    while nums[right] == nums[right-1] and right - 1 > left:
                        right -= 1
                    left += 1
                    right -= 1
                elif nums[i] + nums[left] + nums[right] > 0:
                    right -= 1
                else:
                    left += 1
        return path
# @lc code=end

s = Solution()
s.threeSum([-1,-1,0,1])
