# 给定一个 n 个元素有序的（升序）整型数组 nums 和一个目标值 target ，写一个函数搜索 nums 中的 target，如果目标值存在返回下标，否
# 则返回 -1。 
# 
#  
# 示例 1: 
# 
#  输入: nums = [-1,0,3,5,9,12], target = 9
# 输出: 4
# 解释: 9 出现在 nums 中并且下标为 4
#  
# 
#  示例 2: 
# 
#  输入: nums = [-1,0,3,5,9,12], target = 2
# 输出: -1
# 解释: 2 不存在 nums 中因此返回 -1
#  
# 
#  
# 
#  提示： 
# 
#  
#  你可以假设 nums 中的所有元素是不重复的。 
#  n 将在 [1, 10000]之间。 
#  nums 的每个元素都将在 [-9999, 9999]之间。 
#  
#  Related Topics 数组 二分查找 
#  👍 493 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
# class Solution:
#     def search(self, nums: List[int], target: int) -> int:
#         '''
#         23:02	info
# 			解答成功:
# 			执行耗时:40 ms,击败了59.57% 的Python3用户
# 			内存消耗:15.7 MB,击败了86.36% 的Python3用户
# 		二分查找 -》 左闭右闭
# 		taget 在 [left, right] 区间内，所有要用 left <= right，因为 left == right 是有意义的， 因此 left+1 和 right-1
# 		注意边界条件，当左指针大于右指针时，退出循环，这里的条件和左右指针与mid指针的关系对应
#         '''
#         left, right = 0, len(nums)-1 # 注意左闭右闭，右边有意义，因此要减1
#         while (left <= right):
#             mid = (left + right + 1) // 2
#             if nums[mid] == target:
#                 return mid
#             elif nums[mid] > target:
#                 right = mid - 1
#             else:
#                 left = mid + 1
#         return -1
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        '''
        23:30	info
			解答成功:
			执行耗时:44 ms,击败了31.51% 的Python3用户
			内存消耗:15.7 MB,击败了83.68% 的Python3用户
		二分查找 -》 左闭右开
		taget 在 [left, right) 区间内，所有要用 left < right，因为 left == right 是无意义的， 因此 left+1 和 right
        '''
        left, right = 0, len(nums)   # 注意左闭右开，右边无意义，因此不用减1
        while (left < right):
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] > target:
                right = mid
            else:
                left = mid + 1
        return -1
# leetcode submit region end(Prohibit modification and deletion)
