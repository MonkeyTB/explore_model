# 给你一个由 n 个整数组成的数组 nums ，和一个目标值 target 。请你找出并返回满足下述全部条件且不重复的四元组 [nums[a], nums[b
# ], nums[c], nums[d]] （若两个四元组元素一一对应，则认为两个四元组重复）： 
# 
#  
#  0 <= a, b, c, d < n 
#  a、b、c 和 d 互不相同 
#  nums[a] + nums[b] + nums[c] + nums[d] == target 
#  
# 
#  你可以按 任意顺序 返回答案 。 
# 
#  
# 
#  示例 1： 
# 
#  
# 输入：nums = [1,0,-1,0,-2,2], target = 0
# 输出：[[-2,-1,1,2],[-2,0,0,2],[-1,0,0,1]]
#  
# 
#  示例 2： 
# 
#  
# 输入：nums = [2,2,2,2,2], target = 8
# 输出：[[2,2,2,2]]
#  
# 
#  
# 
#  提示： 
# 
#  
#  1 <= nums.length <= 200 
#  -109 <= nums[i] <= 109 
#  -109 <= target <= 109 
#  
#  Related Topics 数组 双指针 排序 
#  👍 1035 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        '''
        9:11	info
		解答成功:
		执行耗时:712 ms,击败了64.76% 的Python3用户
		内存消耗:15 MB,击败了53.71% 的Python3用户
        注意：
        再三数求和的基础上再加一层循环即可，注意剪枝操作
        '''
        res = []
        n = len(nums)
        nums.sort()
        for i in range(n):
            if i > 0 and nums[i] == nums[i-1]: continue
            for j in range(i+1,n):
                if j > i + 1 and nums[j] == nums[j-1]: continue
                left = j + 1
                right = n - 1
                while left < right:
                    sum = nums[i] + nums[j] + nums[left] + nums[right]
                    if sum > target:
                        right -= 1
                    elif sum < target:
                        left += 1
                    else:
                        res.append([nums[i], nums[j], nums[left], nums[right]])
                        while left < right and nums[left] == nums[left+1]: left += 1
                        while left < right and nums[right] == nums[right-1]: right -= 1
                        left += 1
                        right -= 1
        return res
# leetcode submit region end(Prohibit modification and deletion)
