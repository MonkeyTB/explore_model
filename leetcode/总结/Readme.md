# Sum 问题
+ 给定一个target和一个list数组，指定多个元素之和为target
    + 排序+双指针
    ```python 
    # 15
    from typing import List
    class Solution:
        def threeSum(self, nums: List[int]) -> List[List[int]]:
            nums.sort() # 排序
            path = []
            for i,q in enumerate(nums):
                left = i + 1
                right = len(nums) - 1
                while left < right: # 双指针
                    if -nums[i] == nums[left] + nums[right]:
                        if [nums[i], nums[left], nums[right]] not in path:
                            path.append([nums[i], nums[left], nums[right]])
                    
                        left += 1
                        right -= 1
                    elif nums[i] + nums[left] + nums[right] > 0:
                        right -= 1
                    else:
                        left += 1
            return path
    ```
+ 更进一步，如果list有重复元素，我们要求返回list不能包含重复元素eg:[1,2],[2,1]，该怎么做呢？
    + 基本框架还是排序+双指针，这里有两种方式
        + 第一种就是跳过相同元素
        + 第二种每次添加的时候查看是否在path中，如上面`if [nums[i], nums[left], nums[right]] not in path:`
    ```python
    from typing import List
    class Solution:
        def threeSum(self, nums: List[int]) -> List[List[int]]:
            # 排序+双指针
            nums.sort()
            path = []
            i = 0
            while i < len(nums):
                while i + 1 < len(nums) and (nums[i] == nums[i+1]):
                    i = i + 1
                left = i + 1
                right = len(nums) - 1
                while left < right:
                    if -nums[i] == nums[left] + nums[right]:
                        path.append([nums[i], nums[left], nums[right]])
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
                i += 1
            return path
    ```
        + 当前代码乍一看很合理，排除了重复元素，但忽略了一种情况`[-1,0,1,2,-1,-4]`输出`[[-1,0,1]]`预期输出`[[-1,-1,2],[-1,0,1]]`