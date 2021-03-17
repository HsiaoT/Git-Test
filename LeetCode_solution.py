
# ------------------------------------------------------------
# Array
# in-place
# Time complexity: O(N)
# return the length
def removeDuplicates(nums):
	if not nums: return 0
	i, j = 0, 1
	while j < len(nums):
		if nums[i] != nums[j]:
			i+=1
			nums[i] = nums[j]
		j+=1
	return i+1

# ------------------------------------------------------------
# Array
# method in-place
def rotate(nums:List[int], k:int) -> None:
	def numReverse(start, end):
		while start < end:
			nums[start], nums[end] = nums[end], nums[start]
			start+=1
			end-=1
	k,n = k%len(nums),len(nums)
	if k:
		numReverse(0, n-1)	#將整個陣列進行反轉
		numReverse(0, k-1)	#將前k個數進行反轉
		numReverse(k, n-1)	#將後n-k個數進行反轉

# ------------------------------------------------------------
# Array
# method not in-place
def rotate(nums:List[int], k:int) -> None:
	k,n = k%len(nums), len(nums)
	if K > 0:
		nums[0:n] = nums[-k:]+nums[:-k]

# ------------------------------------------------------------
# Array
def singleNumber(nums:List[int]) -> int:
	res = 0
	for i in nums:
		res ^= i	# XOR, 重複兩次的數字互相做XOR後會變成0
	return res

# ------------------------------------------------------------
def strStr(haystack, needle) -> int:
	for i in range(len(haystack) - len(needle) + 1):
		if haystack[i:i+len(needle)] == needle:
			return i
	return -1

# ------------------------------------------------------------
def reverse(x:int) -> int:
	flag = 1 if x>=0 else -1
	res, x= 0, abs(x)
	while x:
		res = res*10 + x%10
		x = int(x/10)
	res = flag*res
	return res if res < 2**31-1 and res >= -2**31 else 0

# ------------------------------------------------------------
# String
def longestCommonPrefix(strs):
	if not strs: return ""
	s1 = min(strs)			#排序最大和最小在某個index字元相同時,
	s2 = max(strs)			#即代表中間其他字串在此index字元亦相同
	for i,c in enumerate(s1):
		if c != s2[i]:
			return s1[:i]
	return s1

# ------------------------------------------------------------
# String
def myAtoi(str):
	str.strip()
	if not str: return 0
	rstr = re.findall(r"^[+-]?\d+", str)
	if not rstr: return 0

	maxN = math.pow(2,31)-1
	minN = -(math.pow(2,31))
	if int(rstr[0]) > maxN:
		return int(maxN)
	elif int(rstr[0]) < minN:
		return int(minN)
	else:
		return int(rstr[0])

# ------------------------------------------------------------
# DP
def climbStairs(n):
	if n == 1: return 1
	if n == 2: return 2
	s1, s2 = 1, 2		# 走到n階的方法 = 走到n-1階的方法 + 走到n-2階的方法
	for _ in range(n-2):
		s1, s2 = s2, s1+s2
	return s2

# ------------------------------------------------------------
# DP
def maxSubArray(nums: List[int]) -> int::
	l = len(nums)
	if l == 0: return 0

	res = cur = nums[0]			# cur:包含當前這個元素的最大子陣列和
	for i in range(1,l):		# res: 到目前為止的最大子陣列和
		if cur > 0:
			cur += nums[i]
		else:
			cur = nums[i]
		if cur > res:
			res = cur
	return res

# ------------------------------------------------------------
# DP
def maxSubArray(nums: List[int]) -> int:
	result = cur = nums[0]
	for i in range(1, len(nums)):
		cur = max(cur + nums[i], nums[i])
		result = max(result, cur)
	return result

def maxProfit(prices):
	if len(prices) <= 1:
		return 0
	diff = [prices[i+1] - prices[i] for i in range(len(prices)-1)]
	return max(0, self.maxSubArray(diff))

# ------------------------------------------------------------
# Math
def isPowerOfThree(n):
	if n <= 0: return False
	while n!=1:
		if n%3 != 0: return False
		n /= 3
	return True

# ------------------------------------------------------------
# Math
def isPowerOfTwo(n):
	return n>0 and (n&(n-1)) == 0		# and -> boolean, & -> bitwise



# ------------------------------------------------------------
# ------------------------------------------------------------
# ------------------------------------------------------------
# ------------------------------------------------------------
# ------------------------------------------------------------

# ------------------------------------------------------------
# DP
def rob(nums:List[int]) -> int:
	if len(nums) == 0: return 0
	if len(nums) == 1: return nums[0]

	dp = [0 for i in range(len(nums)+1)]
	dp[0] = 0
	dp[1] = nums[0]
	for i in range(2,len(nums)+1):
		dp[i]=max(dp[i-2]+nums[i-1], dp[i-1])
	return dp[-1]

# ------------------------------------------------------------
# string
def reverseString(s):
	start, end = 0, len(s)-1
	while start<end:
		s[start], s[end] = s[end], s[start]
		start += 1
		end -= 1

# ------------------------------------------------------------
# string
def firstUniqChar(s):
	count_map={}
    for c in s:
		count_map[c] = count_map.get(c,0)+1  #return the value of the itme with specified key
	for i, c in enumerate count_map:         #a value to return if specified key doesn't exist
		if count_map[c] == 1:
			return i
	return -1

# ------------------------------------------------------------
# string
def isAnagram(s,t):
	if len(s) != len(t):
		return False
	dic_s = {}
	dic_t = {}
	for c in s:
		dict_s[c] = dict_s.get(c,0)+1
	for c in t:
		dict_t[c] = dict_t[c].get(c,0)+1
	return dict_t == dict_s

# ------------------------------------------------------------
# string
def isPalindrome(s):
	s = "".join(e for e in s if e.isalnum()).lower()
	return s == s[::-1]

# ------------------------------------------------------------
# others
def hammingWeight(n:int) -> int:
	count = 0
	while n:
		n &= (n-1)
		count += 1
	return count

# ------------------------------------------------------------
# others
def hammingDistance(x:int, y:int) -> int:
	count = 0
	while x or y:
		if x&1 != y&1: count+=1
		x >>= 1
		y >>= 1
	return count

# ------------------------------------------------------------
# others
def reverseBit(n:int) -> int:
	result = 0
	for i in range(32):
		result <<= 1
		result |= n&1
		n >>= 1
	return result

# ------------------------------------------------------------
# ------------------------------------------------------------
# ------------------------------------------------------------
# ------------------------------------------------------------
# ------------------------------------------------------------

# ------------------------------------------------------------
# Others, Pascal's triangle
def generate(numRows):
	if numRows == 0: return []
	elif numRows == 1: return [1]
	elif numRows == 2: return [[1],[1,1]]
	else:
		triangle = [[1],[1,1]]
		for i in range(2,numRows):
			triangle.append([])
			triangle[i].append(1)    # most left
			for j in range(1,i):
				triangle[i].append(triangle[i-1][j-1]+triangle[i-1][j])
			triangle[i].append(1)    # most right
		return triangle

# ------------------------------------------------------------
# Others, valid parentheses
def isValid(s):
	bracket_map = {"(":")", "[":"]", "{":"}"}
	open_par = ["(", "[", "{"]
	stack = []
	for i in s:
		if i in open_par: 
			stack.append(i)
		elif stack and i == bracket_map[stack[-1]]:
			stack.pop()
		else:
			return False
	return stack == []

# ------------------------------------------------------------
# Others
def missingNumber(nums):
	n = len(nums)
	return n*(n+1)/2 - sum(nums)

# ------------------------------------------------------------
# math
def FizzBuzz(n):
	res = []
	for i in range(1,n+1):
		cur = ""
		if i%3 == 0: cur += "Fizz"
		if i%5 == 0: cur += "Buzz"
		if not len(cur): cur += str(i)
		res.append(cur)
	return res

# ------------------------------------------------------------
# math
def countPrime(n):
	if n < 3: return 0
	primes = [True] * n
	primes[0] = primes[1] = False
	for i in range(2, int(n**0.5)+1):
		if primes[i]:
			primes[i*i:n:i]=[False]*len(primes[i*i:n:i])
	return sum(primes)

# ------------------------------------------------------------
# math
def romanToInt(s):
	roman = {"I":1, "V":5, "X":10, "L":50, "C":100, "D":500, "M":100}
	words = list(s)
	total = 0
	for index in range(len(words)-1):
		type = 1 if roman[words[index]]>=roman[words[index+1]] else -1
		total += type*roman[words[index]]
	return total + roman[words[len(s)-1]]









nums = [0,0,3,2,1]
def isStrictlyIncrease(nums):

	flag = 1    # first part
	for i in range(1,len(nums)):
		if flag == 1:
			if nums[i-1]>=nums[i]: flag = 0 #second part

		if flag == 0:
			if nums[i-1]<=nums[i]: return False
	return True



