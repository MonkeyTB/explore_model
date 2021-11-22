# 给定一个含有 n 个正整数的数组和一个正整数 target 。 
# 
#  找出该数组中满足其和 ≥ target 的长度最小的 连续子数组 [numsl, numsl+1, ..., numsr-1, numsr] ，并返回其长
# 度。如果不存在符合条件的子数组，返回 0 。 
# 
#  
# 
#  示例 1： 
# 
#  
# 输入：target = 7, nums = [2,3,1,2,4,3]
# 输出：2
# 解释：子数组 [4,3] 是该条件下的长度最小的子数组。
#  
# 
#  示例 2： 
# 
#  
# 输入：target = 4, nums = [1,4,4]
# 输出：1
#  
# 
#  示例 3： 
# 
#  
# 输入：target = 11, nums = [1,1,1,1,1,1,1,1]
# 输出：0
#  
# 
#  
# 
#  提示： 
# 
#  
#  1 <= target <= 109 
#  1 <= nums.length <= 105 
#  1 <= nums[i] <= 105 
#  
# 
#  
# 
#  进阶： 
# 
#  
#  如果你已经实现 O(n) 时间复杂度的解法, 请尝试设计一个 O(n log(n)) 时间复杂度的解法。 
#  
#  Related Topics 数组 二分查找 前缀和 滑动窗口 
#  👍 824 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        '''
        22:26	info
			解答成功:
			执行耗时:36 ms,击败了91.30% 的Python3用户
			内存消耗:16.7 MB,击败了8.58% 的Python3用户
        滑动窗口
        注意是大于等于，最开始做按等于做的，死活不对，一度怀疑题有问题了

        '''
        i = 0
        res = 0
        M = len(nums) + 1
        for j in range(len(nums)):
            res += nums[j]
            while res >= target:
                M = M if M < j - i + 1 else j - i + 1
                res -= nums[i]
                i += 1
        return M if M != len(nums) + 1 else 0
# leetcode submit region end(Prohibit modification and deletion)
