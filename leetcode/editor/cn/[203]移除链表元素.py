# 给你一个链表的头节点 head 和一个整数 val ，请你删除链表中所有满足 Node.val == val 的节点，并返回 新的头节点 。
#  
# 
#  示例 1： 
# 
#  
# 输入：head = [1,2,6,3,4,5,6], val = 6
# 输出：[1,2,3,4,5]
#  
# 
#  示例 2： 
# 
#  
# 输入：head = [], val = 1
# 输出：[]
#  
# 
#  示例 3： 
# 
#  
# 输入：head = [7,7,7,7], val = 7
# 输出：[]
#  
# 
#  
# 
#  提示： 
# 
#  
#  列表中的节点数目在范围 [0, 104] 内 
#  1 <= Node.val <= 50 
#  0 <= val <= 50 
#  
#  Related Topics 递归 链表 
#  👍 740 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeElements(self, head: ListNode, val: int) -> ListNode:
        '''
        22:02	info
			解答成功:
			执行耗时:64 ms,击败了34.39% 的Python3用户
			内存消耗:17.9 MB,击败了82.12% 的Python3用户
        注意：
        定义一个虚得头节点：pre_head = ListNode(next = head)
        判断相等：cur.next 因为加了虚得头节点
        '''
        pre_head = ListNode(next=head)
        cur = pre_head
        while cur.next != None:
            if cur.next.val == val:
                cur.next = cur.next.next
            else:
                cur = cur.next
        return pre_head.next
# leetcode submit region end(Prohibit modification and deletion)
