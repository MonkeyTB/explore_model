# 给你四个整数数组 nums1、nums2、nums3 和 nums4 ，数组长度都是 n ，请你计算有多少个元组 (i, j, k, l) 能满足： 
# 
#  
#  0 <= i, j, k, l < n 
#  nums1[i] + nums2[j] + nums3[k] + nums4[l] == 0 
#  
# 
#  
# 
#  示例 1： 
# 
#  
# 输入：nums1 = [1,2], nums2 = [-2,-1], nums3 = [-1,2], nums4 = [0,2]
# 输出：2
# 解释：
# 两个元组如下：
# 1. (0, 0, 0, 1) -> nums1[0] + nums2[0] + nums3[0] + nums4[1] = 1 + (-2) + (-1)
#  + 2 = 0
# 2. (1, 1, 0, 0) -> nums1[1] + nums2[1] + nums3[0] + nums4[0] = 2 + (-1) + (-1)
#  + 0 = 0
#  
# 
#  示例 2： 
# 
#  
# 输入：nums1 = [0], nums2 = [0], nums3 = [0], nums4 = [0]
# 输出：1
#  
# 
#  
# 
#  提示： 
# 
#  
#  n == nums1.length 
#  n == nums2.length 
#  n == nums3.length 
#  n == nums4.length 
#  1 <= n <= 200 
#  -228 <= nums1[i], nums2[i], nums3[i], nums4[i] <= 228 
#  
#  Related Topics 数组 哈希表 
#  👍 456 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def fourSumCount(self, nums1: List[int], nums2: List[int], nums3: List[int], nums4: List[int]) -> int:
        '''
        8:39	info
		解答成功:
		执行耗时:816 ms,击败了7.26% 的Python3用户
		内存消耗:15.2 MB,击败了61.95% 的Python3用户
        解析：
        题目要求统计和为0的个数，因此统计1和2和的种类，再用3和4和统计个数
        '''
        dict_pre = {}
        for i in nums1:
            for x in nums2:
                if i+x in dict_pre.keys():
                    dict_pre[(i+x)] += 1
                else:
                    dict_pre[(i + x)] = 1
        count = 0
        for j in nums3:
            for y in nums4:
                if 0-(y+j) in dict_pre.keys():
                    count += dict_pre[0-(y+j)]
        return count
# leetcode submit region end(Prohibit modification and deletion)
