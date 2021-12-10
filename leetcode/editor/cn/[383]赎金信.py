# 为了不在赎金信中暴露字迹，从杂志上搜索各个需要的字母，组成单词来表达意思。 
# 
#  给你一个赎金信 (ransomNote) 字符串和一个杂志(magazine)字符串，判断 ransomNote 能不能由 magazines 里面的字符
# 构成。 
# 
#  如果可以构成，返回 true ；否则返回 false 。 
# 
#  magazine 中的每个字符只能在 ransomNote 中使用一次。 
# 
#  
# 
#  示例 1： 
# 
#  
# 输入：ransomNote = "a", magazine = "b"
# 输出：false
#  
# 
#  示例 2： 
# 
#  
# 输入：ransomNote = "aa", magazine = "ab"
# 输出：false
#  
# 
#  示例 3： 
# 
#  
# 输入：ransomNote = "aa", magazine = "aab"
# 输出：true
#  
# 
#  
# 
#  提示： 
# 
#  
#  1 <= ransomNote.length, magazine.length <= 105 
#  ransomNote 和 magazine 由小写英文字母组成 
#  
#  Related Topics 哈希表 字符串 计数 
#  👍 250 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        '''
        8:39	info
		解答成功:
		执行耗时:92 ms,击败了14.57% 的Python3用户
		内存消耗:15 MB,击败了90.85% 的Python3用户
        方法：
        先保存magazine的所有数据出现次数，再去便利ransomNote的数据
        '''
        dict_ran = {}
        for r in magazine:
            if r in dict_ran.keys():
                dict_ran[r] += 1
            else:
                dict_ran[r] = 1
        for m in ransomNote:
            if m in dict_ran.keys() and dict_ran[m] > 0:
                dict_ran[m] -= 1
            else:
                return False
        return True
# leetcode submit region end(Prohibit modification and deletion)
