# 给定 s 和 t 两个字符串，当它们分别被输入到空白的文本编辑器后，请你判断二者是否相等。# 代表退格字符。 
# 
#  如果相等，返回 true ；否则，返回 false 。 
# 
#  注意：如果对空文本输入退格字符，文本继续为空。 
# 
#  
# 
#  示例 1： 
# 
#  
# 输入：s = "ab#c", t = "ad#c"
# 输出：true
# 解释：S 和 T 都会变成 “ac”。
#  
# 
#  示例 2： 
# 
#  
# 输入：s = "ab##", t = "c#d#"
# 输出：true
# 解释：s 和 t 都会变成 “”。
#  
# 
#  示例 3： 
# 
#  
# 输入：s = "a##c", t = "#a#c"
# 输出：true
# 解释：s 和 t 都会变成 “c”。
#  
# 
#  示例 4： 
# 
#  
# 输入：s = "a#c", t = "b"
# 输出：false
# 解释：s 会变成 “c”，但 t 仍然是 “b”。 
# 
#  
# 
#  提示： 
# 
#  
#  1 <= s.length, t.length <= 200 
#  s 和 t 只含有小写字母以及字符 '#' 
#  
# 
#  
# 
#  进阶： 
# 
#  
#  你可以用 O(N) 的时间复杂度和 O(1) 的空间复杂度解决该问题吗？ 
#  
# 
#  
#  Related Topics 栈 双指针 字符串 模拟 
#  👍 331 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    '''
    22:26	info
			解答成功:
			执行耗时:32 ms,击败了68.27% 的Python3用户
			内存消耗:14.7 MB,击败了99.00% 的Python3用户
	栈方式：
	最简单也最容易思考
    '''
    def help(self, s):
        du = []
        for i in s:
            if i != '#': du.append(i)
            if i == '#' and len(du) > 0: du.pop(-1)
        return du
    def backspaceCompare(self, s, t) :
        return self.help(s) == self.help(t)


# leetcode submit region end(Prohibit modification and deletion)
