# 给你一个字符串数组 words ，请你找出所有在 words 的每个字符串中都出现的共用字符（ 包括重复字符），并以数组形式返回。你可以按 任意顺序 返回答
# 案。
#  
# 
#  示例 1： 
# 
#  
# 输入：words = ["bella","label","roller"]
# 输出：["e","l","l"]
#  
# 
#  示例 2： 
# 
#  
# 输入：words = ["cool","lock","cook"]
# 输出：["c","o"]
#  
# 
#  
# 
#  提示： 
# 
#  
#  1 <= words.length <= 100 
#  1 <= words[i].length <= 100 
#  words[i] 由小写英文字母组成 
#  
#  Related Topics 数组 哈希表 字符串 
#  👍 246 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def commonChars(self, words):
        list_num = [[0 for i in range(26)] for j in range(len(words))]
        # list_num = [[0 for _ in range(26)] for _ in range(len(words))]
        for i,word in enumerate(words):
            for j,w in enumerate(word):
                list_num[i][ord(w)-97] += 1
        list_num = min(list_num)
        res = []
        for i in range(26):
            if list_num[i] > 0:
                res.append([chr(ord('a')+i)]*list_num[i])
        return [j for i in res for j in i]

ob = Solution()
ob.commonChars(["bella","label","roller"])

# leetcode submit region end(Prohibit modification and deletion)
