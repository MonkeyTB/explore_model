# 给你二叉树的根节点 root 和一个整数目标和 targetSum ，找出所有 从根节点到叶子节点 路径总和等于给定目标和的路径。 
# 
#  叶子节点 是指没有子节点的节点。 
# 
#  
#  
#  
#  
#  
# 
#  示例 1： 
#  
#  
# 输入：root = [5,4,8,11,null,13,4,7,2,null,null,5,1], targetSum = 22
# 输出：[[5,4,11,2],[5,8,4,5]]
#  
# 
#  示例 2： 
#  
#  
# 输入：root = [1,2,3], targetSum = 5
# 输出：[]
#  
# 
#  示例 3： 
# 
#  
# 输入：root = [1,2], targetSum = 0
# 输出：[]
#  
# 
#  
# 
#  提示： 
# 
#  
#  树中节点总数在范围 [0, 5000] 内 
#  -1000 <= Node.val <= 1000 
#  -1000 <= targetSum <= 1000 
#  
# 
#  Related Topics 树 深度优先搜索 回溯 二叉树 👍 809 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
import collections


class Solution:
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> List[List[int]]:
        if root == None:
            return []
        ans = []
        queue = collections.deque([(root, root.val, [root.val])])
        while queue:
            node, temp_sum, temp_ans = queue.popleft()
            if not node.left and not node.right:
                if temp_sum == targetSum:
                    ans.append(temp_ans)
            if node.left:
                queue.append((node.left, temp_sum+node.left.val, temp_ans+[node.left.val]))
            if node.right:
                queue.append((node.right, temp_sum+node.right.val, temp_ans+[node.right.val]))
        return ans
# leetcode submit region end(Prohibit modification and deletion)
