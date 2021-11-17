# 给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。 
# 
#  示例: 
# 
#  输入: [0,1,0,3,12]
# 输出: [1,3,12,0,0] 
# 
#  说明: 
# 
#  
#  必须在原数组上操作，不能拷贝额外的数组。 
#  尽量减少操作次数。 
#  
#  Related Topics 数组 双指针 
#  👍 1302 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        '''
        
        23:19	info
			解答成功:
			执行耗时:44 ms,击败了55.57% 的Python3用户
			内存消耗:15.4 MB,击败了38.14% 的Python3用户
		双指针
		慢指针先找到0值，永远指向0值，fast非0值和slow互换
        '''
        slow, fast = 0, 0
        length = len(nums)
        while slow < length:
            if nums[slow] == 0:
                break
            slow += 1
        fast = slow + 1
        while fast < length:
            if nums[fast] != 0:
                nums[slow], nums[fast] = nums[fast], nums[slow]
                slow += 1
            fast += 1

# leetcode submit region end(Prohibit modification and deletion)
