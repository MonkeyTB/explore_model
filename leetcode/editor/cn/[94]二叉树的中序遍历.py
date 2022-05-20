# 给定一个二叉树的根节点 root ，返回 它的 中序 遍历 。 
# 
#  
# 
#  示例 1： 
# 
#  
# 输入：root = [1,null,2,3]
# 输出：[1,3,2]
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
#  树中节点数目在范围 [0, 100] 内 
#  -100 <= Node.val <= 100 
#  
# 
#  
# 
#  进阶: 递归算法很简单，你可以通过迭代算法完成吗？ 
#  Related Topics 栈 树 深度优先搜索 二叉树 
#  👍 1415 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
'''
# 递归
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        res = []
        def traversal(root):
            if root == None:
                return
            traversal(root.left)
            res.append(root.val)
            traversal(root.right)
        traversal(root)
        return res
'''
# 迭代
class TreeNode:
    def __init__(self, value):
        self.val = value
        self.left = None
        self.right = None
class Solution:
    def inorderTraversal(self, root:Optional[TreeNode]) -> List[int]:
        if not root:
            return []
        stack = []
        cru = root
        res = []
        while cru or stack:
            if cru:# 先迭代访问最底层的左子树结点
                stack.append(cru)
                cru = cru.left
            else:# 到达最左结点后处理栈顶结点
                cru = stack.pop()
                res.append(cru.val)
                cru = cru.right # 取栈顶元素右结点
        return res
# leetcode submit region end(Prohibit modification and deletion)
