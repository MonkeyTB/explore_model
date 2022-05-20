# 给你一棵二叉树的根节点 root ，返回其节点值的 后序遍历 。 
# 
#  
# 
#  示例 1： 
# 
#  
# 输入：root = [1,null,2,3]
# 输出：[3,2,1]
#  
# 
#  示例 2： 
# 
#  
# 输入：root = []
# 输出：[]
#  
# 
#  示例 3： 
# 
#  
# 输入：root = [1]
# 输出：[1]
#  
# 
#  
# 
#  提示： 
# 
#  
#  树中节点的数目在范围 [0, 100] 内 
#  -100 <= Node.val <= 100 
#  
# 
#  
# 
#  进阶：递归算法很简单，你可以通过迭代算法完成吗？ 
#  Related Topics 栈 树 深度优先搜索 二叉树 
#  👍 831 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
'''
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        res = []
        def traversal(root):
            if root == None:
                return
            traversal(root.left) # 左
            traversal(root.right) # 右
            res.append(root.val) # 中
        traversal(root)
        return res
'''
# 迭代法
class TreeNoe:
    def __init__(self, value):
        self.val = value
        self.left = None
        self.right = None
class Solution:
    def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []
        res = []
        stack = [root]

        while stack:
            node = stack.pop()
            res.append(node.val)  # 中结点先处理
            if node.left:
                stack.append(node.left) # 左结点先处理
            if node.right:
                stack.append(node.right) # 右结点先处理
        return res[::-1]
# leetcode submit region end(Prohibit modification and deletion)
