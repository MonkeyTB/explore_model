# 给你一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？请你找出所有和为 0 且不重
# 复的三元组。 
# 
#  注意：答案中不可以包含重复的三元组。 
# 
#  
# 
#  示例 1： 
# 
#  
# 输入：nums = [-1,0,1,2,-1,-4]
# 输出：[[-1,-1,2],[-1,0,1]]
#  
# 
#  示例 2： 
# 
#  
# 输入：nums = []
# 输出：[]
#  
# 
#  示例 3： 
# 
#  
# 输入：nums = [0]
# 输出：[]
#  
# 
#  
# 
#  提示： 
# 
#  
#  0 <= nums.length <= 3000 
#  -105 <= nums[i] <= 105 
#  
#  Related Topics 数组 双指针 排序 
#  👍 4072 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        '''
        9:21	info
		解答成功:
		执行耗时:476 ms,击败了92.39% 的Python3用户
		内存消耗:17.6 MB,击败了41.04% 的Python3用户
        方法:
        1.排序
        2.三指针,便利
            大于0,右指针移动
            小于0,左指针移动
            等于0,添加到结果,判断移动左指针还是右指针
        '''
        nums.sort()
        res = []
        n = len(nums)
        for i in range(len(nums)):
            if nums[i] > 0: break
            if i >= 1 and nums[i] == nums[i-1]: continue
            left = i + 1
            right = n - 1
            while left < right:
                tag = nums[i] + nums[left] + nums[right]
                if tag == 0:
                    res.append([nums[i], nums[left], nums[right]])
                    while left != right and nums[left] == nums[left+1]:
                        left += 1
                    while left != right and nums[right] == nums[right-1]:
                        right -= 1
                    left += 1
                    right -= 1
                elif tag > 0:
                    right -= 1
                else:
                    left += 1
        return res
# leetcode submit region end(Prohibit modification and deletion)
