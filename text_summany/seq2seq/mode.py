def minDistance( word1: str, word2: str) -> int:
    m = len(word1) + 1
    n = len(word2) + 1
    dp = [[0]*n for j in range(m)]
    print(len(dp), len(dp[0]))
    for i in range(m):
        dp[i][0] = i
    for j in range(n):
        dp[0][j] = j
    for i,w1 in enumerate(word1):
        for j,w2 in enumerate(word2):
            if w1 == w2:
                dp[i+1][j+1] = dp[i][j] 
            else:
                dp[i+1][j+1] = min(dp[i][j+1], dp[i+1][j], dp[i][j]) + 1
    return dp
minDistance(word1 = "horse", word2 = "ros")