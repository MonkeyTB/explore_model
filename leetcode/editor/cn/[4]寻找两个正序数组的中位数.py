# 给定两个大小分别为 m 和 n 的正序（从小到大）数组 nums1 和 nums2。请你找出并返回这两个正序数组的 中位数 。 
# 
#  算法的时间复杂度应该为 O(log (m+n)) 。 
# 
#  
# 
#  示例 1： 
# 
#  
# 输入：nums1 = [1,3], nums2 = [2]
# 输出：2.00000
# 解释：合并数组 = [1,2,3] ，中位数 2
#  
# 
#  示例 2： 
# 
#  
# 输入：nums1 = [1,2], nums2 = [3,4]
# 输出：2.50000
# 解释：合并数组 = [1,2,3,4] ，中位数 (2 + 3) / 2 = 2.5
#  
# 
#  示例 3： 
# 
#  
# 输入：nums1 = [0,0], nums2 = [0,0]
# 输出：0.00000
#  
# 
#  示例 4： 
# 
#  
# 输入：nums1 = [], nums2 = [1]
# 输出：1.00000
#  
# 
#  示例 5： 
# 
#  
# 输入：nums1 = [2], nums2 = []
# 输出：2.00000
#  
# 
#  
# 
#  提示： 
# 
#  
#  nums1.length == m 
#  nums2.length == n 
#  0 <= m <= 1000 
#  0 <= n <= 1000 
#  1 <= m + n <= 2000 
#  -10⁶ <= nums1[i], nums2[i] <= 10⁶ 
#  
#  Related Topics 数组 二分查找 分治 👍 4641 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        '''
        			解答成功:
			执行耗时:48 ms,击败了40.60% 的Python3用户
			内存消耗:15.2 MB,击败了23.19% 的Python3用户

        将两个有序数组合并到一个有序数组，取数返回
        '''
        res = []
        i, j, len1, len2 = 0, 0, len(nums1), len(nums2)
        while i < len1 and j < len2:
            if nums1[i] > nums2[j]:
                res.append(nums2[j])
                j += 1
            else:
                res.append(nums1[i])
                i += 1
        if i < len1: res += nums1[i:len1]
        if j < len2: res += nums2[j:len2]
        if len(res) % 2 == 0:
            return (res[len(res)//2] + res[len(res)//2-1] ) / 2.
        else:
            return  res[len(res)//2]


# leetcode submit region end(Prohibit modification and deletion)
