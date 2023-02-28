# Sum 问题
## 给定一个target和一个list数组，指定多个元素之和为target leetcode15
+ 排序+双指针
    ```python 
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
    明晚再改
    ```python
    # 正确方式
    from typing import List
    class Solution:
        def threeSum(self, nums: List[int]) -> List[List[int]]:
            # 排序+双指针
            nums.sort()
            path = []
            for i in range(len(nums)):
                # 去重逻辑[-1,-1,2]
                if i > 0 and nums[i] == nums[i-1]: # 注意这里
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
    ```
$\color{blue}{值得注意的是，判断是否存在（第一种6452 ms）和跳过重复元素（第二种1616 ms）时间差距很大}$
## 改变一下，求三数之和最接近 leetcode 16
+ 
    ```python
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
    ```
## 升级版，求四树之和 leetcode 18
+ 还是用上面的方法，两层for循环+双指针
    ```python
    from typing import List
    class Solution:
        def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
            nums.sort()
            path = []
            for i in range(len(nums)):
                if i > 0 and nums[i] == nums[i-1]:
                    continue
                for j in range(i+1, len(nums)):
                    if j > i+1 and nums[j] == nums[j-1]: # 注意这里是j>i+1
                        continue
                    left, right = j + 1, len(nums) - 1
                    while left < right:
                        if nums[i] + nums[j] + nums[left] + nums[right] == target:
                            path.append([nums[i], nums[j], nums[left], nums[right]])
                            while left + 1 < right and nums[left] == nums[left+1]: left += 1
                            while right - 1 > left and nums[right] == nums[right-1]: right -= 1
                            left += 1
                            right -= 1
                        elif nums[i] + nums[j] + nums[left] + nums[right] > target:
                            right -= 1
                        else:
                            left += 1
            return path
    ```