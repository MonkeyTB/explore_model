# 给定两个字符串 s 和 t ，编写一个函数来判断 t 是否是 s 的字母异位词。 
# 
#  注意：若 s 和 t 中每个字符出现的次数都相同，则称 s 和 t 互为字母异位词。 
# 
#  
# 
#  示例 1: 
# 
#  
# 输入: s = "anagram", t = "nagaram"
# 输出: true
#  
# 
#  示例 2: 
# 
#  
# 输入: s = "rat", t = "car"
# 输出: false 
# 
#  
# 
#  提示: 
# 
#  
#  1 <= s.length, t.length <= 5 * 104 
#  s 和 t 仅包含小写字母 
#  
# 
#  
# 
#  进阶: 如果输入字符串包含 unicode 字符怎么办？你能否调整你的解法来应对这种情况？ 
#  Related Topics 哈希表 字符串 排序 
#  👍 459 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        '''
        22:40	info
			解答成功:
			执行耗时:40 ms,击败了88.41% 的Python3用户
			内存消耗:14.9 MB,击败了96.80% 的Python3用户
        :param s:
        :param t:
        :return:
        '''
        dict_num = {}
        for i in s:
            if i in dict_num.keys():
                dict_num[i] += 1
            else:
                dict_num[i] = 1
        for j in t:
            if j not in dict_num.keys():
                return False
            else:
                dict_num[j] -= 1
        for key,value in dict_num.items():
            if value != 0:
                return False
        return True
# leetcode submit region end(Prohibit modification and deletion)
