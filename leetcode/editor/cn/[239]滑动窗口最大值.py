# 给你一个整数数组 nums，有一个大小为 k 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 k 个数字。滑动窗口每次只向右移动一位
# 。 
# 
#  返回滑动窗口中的最大值。 
# 
#  
# 
#  示例 1： 
# 
#  
# 输入：nums = [1,3,-1,-3,5,3,6,7], k = 3
# 输出：[3,3,5,5,6,7]
# 解释：
# 滑动窗口的位置                最大值
# ---------------               -----
# [1  3  -1] -3  5  3  6  7       3
#  1 [3  -1  -3] 5  3  6  7       3
#  1  3 [-1  -3  5] 3  6  7       5
#  1  3  -1 [-3  5  3] 6  7       5
#  1  3  -1  -3 [5  3  6] 7       6
#  1  3  -1  -3  5 [3  6  7]      7
#  
# 
#  示例 2： 
# 
#  
# 输入：nums = [1], k = 1
# 输出：[1]
#  
# 
#  示例 3： 
# 
#  
# 输入：nums = [1,-1], k = 1
# 输出：[1,-1]
#  
# 
#  示例 4： 
# 
#  
# 输入：nums = [9,11], k = 2
# 输出：[11]
#  
# 
#  示例 5： 
# 
#  
# 输入：nums = [4,-2], k = 2
# 输出：[4] 
# 
#  
# 
#  提示： 
# 
#  
#  1 <= nums.length <= 105 
#  -104 <= nums[i] <= 104 
#  1 <= k <= nums.length 
#  
#  Related Topics 队列 数组 滑动窗口 单调队列 堆（优先队列） 
#  👍 1326 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
# class Solution:
#     '''
#     O(n^2)
#     '''
#     def findMax(self, s:List[int]):
#         maxValue = -float('inf')
#         for i in s:
#             if i > maxValue:
#                 maxValue = i
#         return maxValue
#
#     def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
#         if k == 1: return nums
#         res = []
#         for i in range(len(nums) - k + 1):
#             mid = nums[i:i+k]
#             res.append(self.findMax(mid))
#         return res
class MyQueue:  # 单调队列（从大到小
    def __init__(self):
        self.queue = []  # 使用list来实现单调队列

    # 每次弹出的时候，比较当前要弹出的数值是否等于队列出口元素的数值，如果相等则弹出。
    # 同时pop之前判断队列当前是否为空。
    def pop(self, value):
        if self.queue and value == self.queue[0]:
            self.queue.pop(0)  # list.pop()时间复杂度为O(n),这里可以使用collections.deque()

    # 如果push的数值大于入口元素的数值，那么就将队列后端的数值弹出，直到push的数值小于等于队列入口元素的数值为止。
    # 这样就保持了队列里的数值是单调从大到小的了。
    def push(self, value):
        while self.queue and value > self.queue[-1]:
            self.queue.pop()
        self.queue.append(value)

    # 查询当前队列里的最大值 直接返回队列前端也就是front就可以了。
    def front(self):
        return self.queue[0]


class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        que = MyQueue()
        result = []
        for i in range(k):  # 先将前k的元素放进队列
            que.push(nums[i])
        result.append(que.front())  # result 记录前k的元素的最大值
        for i in range(k, len(nums)):
            que.pop(nums[i - k])  # 滑动窗口移除最前面元素
            que.push(nums[i])  # 滑动窗口前加入最后面的元素
            result.append(que.front())  # 记录对应的最大值
        return result
# leetcode submit region end(Prohibit modification and deletion)
