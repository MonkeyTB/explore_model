# 给定一个链表，两两交换其中相邻的节点，并返回交换后的链表。 
# 
#  你不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。 
# 
#  
# 
#  示例 1： 
# 
#  
# 输入：head = [1,2,3,4]
# 输出：[2,1,4,3]
#  
# 
#  示例 2： 
# 
#  
# 输入：head = []
# 输出：[]
#  
# 
#  示例 3： 
# 
#  
# 输入：head = [1]
# 输出：[1]
#  
# 
#  
# 
#  提示： 
# 
#  
#  链表中节点的数目在范围 [0, 100] 内 
#  0 <= Node.val <= 100 
#  
# 
#  
# 
#  进阶：你能在不修改链表节点值的情况下解决这个问题吗?（也就是说，仅修改节点本身。） 
#  Related Topics 递归 链表 
#  👍 1125 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def swapPairs(self, head: ListNode) -> ListNode:
        '''
        23:46	info
			解答成功:
			执行耗时:32 ms,击败了66.17% 的Python3用户
			内存消耗:15 MB,击败了44.06% 的Python3用户
        :param head:
        :return:
        '''
        res = ListNode(next=head)
        pre = res
        while pre.next and pre.next.next:
            cur = pre.next
            temp = cur.next

            cur.next = temp.next
            temp.next = cur
            pre.next = temp


            pre = pre.next.next

        return res.next

# leetcode submit region end(Prohibit modification and deletion)
