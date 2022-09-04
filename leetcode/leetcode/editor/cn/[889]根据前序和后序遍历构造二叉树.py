# 给定两个整数数组，preorder 和 postorder ，其中 preorder 是一个具有 无重复 值的二叉树的前序遍历，postorder 是同一棵
# 树的后序遍历，重构并返回二叉树。 
# 
#  如果存在多个答案，您可以返回其中 任何 一个。 
# 
#  
# 
#  示例 1： 
# 
#  
# 
#  
# 输入：preorder = [1,2,4,5,3,6,7], postorder = [4,5,2,6,7,3,1]
# 输出：[1,2,3,4,5,6,7]
#  
# 
#  示例 2: 
# 
#  
# 输入: preorder = [1], postorder = [1]
# 输出: [1]
#  
# 
#  
# 
#  提示： 
# 
#  
#  1 <= preorder.length <= 30 
#  1 <= preorder[i] <= preorder.length 
#  preorder 中所有值都 不同 
#  postorder.length == preorder.length 
#  1 <= postorder[i] <= postorder.length 
#  postorder 中所有值都 不同 
#  保证 preorder 和 postorder 是同一棵二叉树的前序遍历和后序遍历 
#  
# 
#  Related Topics 树 数组 哈希表 分治 二叉树 👍 268 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def constructFromPrePost(self, preorder: List[int], postorder: List[int]) -> Optional[TreeNode]:

        def creatTree(preorder, postorder, length):
            if length == 0:
                return
            node = TreeNode(preorder[0])
            if length == 1:
                return node
            k = 0
            while preorder[1] != postorder[k]:
                k += 1
            node.left = creatTree(preorder[1:k+2], postorder[:k+1], k+1)
            node.right = creatTree(preorder[k+2:], postorder[k+1:-1], length-k-2)
            return node
        return creatTree(preorder, postorder, len(preorder))
# leetcode submit region end(Prohibit modification and deletion)
