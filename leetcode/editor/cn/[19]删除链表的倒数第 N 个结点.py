# 给你一个链表，删除链表的倒数第 n 个结点，并且返回链表的头结点。 
# 
#  
# 
#  示例 1： 
# 
#  
# 输入：head = [1,2,3,4,5], n = 2
# 输出：[1,2,3,5]
#  
# 
#  示例 2： 
# 
#  
# 输入：head = [1], n = 1
# 输出：[]
#  
# 
#  示例 3： 
# 
#  
# 输入：head = [1,2], n = 1
# 输出：[1]
#  
# 
#  
# 
#  提示： 
# 
#  
#  链表中结点的数目为 sz 
#  1 <= sz <= 30 
#  0 <= Node.val <= 100 
#  1 <= n <= sz 
#  
# 
#  
# 
#  进阶：你能尝试使用一趟扫描实现吗？ 
#  Related Topics 链表 双指针 
#  👍 1664 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        '''
        22:05	info
			解答成功:
			执行耗时:40 ms,击败了18.83% 的Python3用户
			内存消耗:14.9 MB,击败了73.57% 的Python3用户
        快慢指针法
        注意:
        slow, fast = head_dummy, head_dummy 这种可以跑通
        slow, fast = ListNode(next = head), ListNode(next = head) 这种不可以
        不明白为啥,python链表用得不多,感觉都是定义几个指针,让next指向head节点,但是结果显示看来是由区别的
        '''
        head_dummy = ListNode(next = head)
        slow, fast = head_dummy, head_dummy
        # slow, fast = ListNode(next = head), ListNode(next = head)
        while n != 0:
            fast = fast.next
            n -= 1
        while fast.next:
            slow = slow.next
            fast = fast.next
        slow.next = slow.next.next
        return head_dummy.next

# leetcode submit region end(Prohibit modification and deletion)
