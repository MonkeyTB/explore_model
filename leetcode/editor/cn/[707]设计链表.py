# 设计链表的实现。您可以选择使用单链表或双链表。单链表中的节点应该具有两个属性：val 和 next。val 是当前节点的值，next 是指向下一个节点的指针
# /引用。如果要使用双向链表，则还需要一个属性 prev 以指示链表中的上一个节点。假设链表中的所有节点都是 0-index 的。 
# 
#  在链表类中实现这些功能： 
# 
#  
#  get(index)：获取链表中第 index 个节点的值。如果索引无效，则返回-1。 
#  addAtHead(val)：在链表的第一个元素之前添加一个值为 val 的节点。插入后，新节点将成为链表的第一个节点。 
#  addAtTail(val)：将值为 val 的节点追加到链表的最后一个元素。 
#  addAtIndex(index,val)：在链表中的第 index 个节点之前添加值为 val 的节点。如果 index 等于链表的长度，则该节点将附加
# 到链表的末尾。如果 index 大于链表长度，则不会插入节点。如果index小于0，则在头部插入节点。 
#  deleteAtIndex(index)：如果索引 index 有效，则删除链表中的第 index 个节点。 
#  
# 
#  
# 
#  示例： 
# 
#  MyLinkedList linkedList = new MyLinkedList();
# linkedList.addAtHead(1);
# linkedList.addAtTail(3);
# linkedList.addAtIndex(1,2);   //链表变为1-> 2-> 3
# linkedList.get(1);            //返回2
# linkedList.deleteAtIndex(1);  //现在链表是1-> 3
# linkedList.get(1);            //返回3
#  
# 
#  
# 
#  提示： 
# 
#  
#  所有val值都在 [1, 1000] 之内。 
#  操作次数将在 [1, 1000] 之内。 
#  请不要使用内置的 LinkedList 库。 
#  
#  Related Topics 设计 链表 
#  👍 306 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Node:
    def __init__(self, val):
        self.val = val
        self.next = None
class MyLinkedList:

    def __init__(self):
        self._head = Node(0)
        self._count = 0


    def get(self, index: int) -> int:
        if 0 <= index <= self._count:
            node = self._head
            for _ in range(index + 1):
                node = node.next
            return node.val
        else:
            return -1


    def addAtHead(self, val: int) -> None:
        self.addAtIndex(0, val)


    def addAtTail(self, val: int) -> None:
        self.addAtIndex(self._count, val)


    def addAtIndex(self, index: int, val: int) -> None:
        if index > self._count:
            return
        elif index < 0:
            index = 0
        self._count += 1
        add_node = Node(val)
        pre, cur = None, self._head
        for _ in range(index + 1):
            pre, cur = cur, cur.next
        else:
            pre.next, add_node.next = add_node, cur


    def deleteAtIndex(self, index: int) -> None:
        if index < 0 or index > self._count:
            return
        self._count -= 1
        pre, cur = None, self._head
        for _ in range(index + 1):
            pre, cur = cur, cur.next
        pre.next, cur = cur.next, None


# Your MyLinkedList object will be instantiated and called as such:
# obj = MyLinkedList()
# param_1 = obj.get(index)
# obj.addAtHead(val)
# obj.addAtTail(val)
# obj.addAtIndex(index,val)
# obj.deleteAtIndex(index)
# leetcode submit region end(Prohibit modification and deletion)
