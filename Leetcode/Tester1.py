import collections
import math


# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


def toHex(num):
    """
    :type num: int
    :rtype: str
    """
    s = "abcdef"
    if num == 0:
        return '0'
    if num < 0:
        num += 2 ** 32
    res = []
    while num:
        d = num % 16
        res.append(str(d) if d < 9 else s[d - 10])
        num //= 16
    return ''.join(res[::-1])  # if num else '0'


def longestPalindrome(s):
    """
    :type s: str
    :rtype: int
    """
    dict = {}
    for c in s:
        if c not in dict:
            dict[c] = 1
        else:
            dict[c] += 1
    print(dict)
    length = 0
    f = False
    for num in dict.values():
        if num % 2 == 0:
            length += num
        else:
            f = True
            if num > 1:
                length += num - 1

    return length + 1 if f else length


def addStrings(num1, num2):
    """
    :type num1: str
    :type num2: str
    :rtype: str
    """
    n = len(num1)
    m = len(num2)
    l = 0
    sum = 0
    while (n > 0 and m > 0):
        sum += (int(num1[n - 1]) + int(num2[m - 1])) * 10 ** l
        l += 1
        n -= 1
        m -= 1

    if n == 0 and m != 0:
        sum += int(num2[:m]) * 10 ** l
    elif m == 0 and n != 0:
        sum += int(num1[:n]) * 10 ** l
    else:
        pass
    return str(sum)


def countSegments(s):
    """
    :type s: str
    :rtype: int
    """
    n = 0
    fl = False
    for c in s:
        if c != ' ':
            print(c)
            fl = True
        if c == ' ' and fl:
            print(c)
            n += 1
            fl = False
    if fl:
        n += 1
    return n


def arrangeCoins(n):
    """
    :type n: int
    :rtype: int
    """
    l = 0
    while (n >= l):
        n -= l
        l += 1
    return l - 1


def findDisappearedNumbers(nums):
    """
    :type nums: List[int]
    :rtype: List[int]
    """
    myset = set(nums)
    mylist = []
    for i in range(len(nums)):
        if (i + 1) not in myset:
            mylist.append(i + 1)

    return mylist


def findContentChildren(g, s):
    """
    :type g: List[int]
    :type s: List[int]
    :rtype: int
    """
    g.sort()
    s.sort()
    i = len(g) - 1
    j = len(s) - 1
    count = 0
    while i >= 0 and j >= 0:
        if s[j] >= g[i]:
            count += 1
            i -= 1
            j -= 1
        else:
            i -= 1
    return count


def constructRectangle(area):
    """
    :type area: int
    :rtype: List[int]
    """
    mid = int(math.sqrt(area))
    while mid > 0:
        if area % mid == 0:
            return [int(area / mid), mid]
        mid -= 1


def findPoisonedDuration(timeSeries, duration):
    """
    :type timeSeries: List[int]
    :type duration: int
    :rtype: int
    """
    total = 0
    l = len(timeSeries)
    for i in range(l):
        t0 = timeSeries[i]
        end = t0 + duration - 1
        if i < l - 1:
            t1 = timeSeries[i + 1]
            if end >= t1:
                total += t1 - t0
            else:
                total += end - t0 + 1
        else:
            total += end - t0 + 1
    return total


def nextGreaterElement(nums1, nums2):
    """
    :type nums1: List[int]
    :type nums2: List[int]
    :rtype: List[int]
    """
    dict = {}
    arr = []

    for i in range(len(nums2)):
        dict[nums2[i]] = i
    for x in nums1:
        start = dict[x]
        l2 = len(nums2)
        for j in range(start, l2):
            if nums2[j] > x:
                arr.append(nums2[j])
                break
        if j == l2 - 1 and nums2[l2 - 1] <= x:
            arr.append(-1)
    return arr


def findWords(words):
    """
    :type words: List[str]
    :rtype: List[str]
    """
    arr = []
    s1 = set("qwertyuiop")
    s2 = set("asdfghjkl")
    s3 = set("zxcvbnm")
    for word in words:
        t1 = True
        t2 = True
        t3 = True
        for c in word.lower():
            if c not in s1:
                t1 = False
            if c not in s2:
                t2 = False
            if c not in s3:
                t3 = False
            if not t1 and not t2 and not t3:
                break
        if t1 or t2 or t3:
            arr.append(word)
    return arr


def convertToBase7(num):
    """
    :type num: int
    :rtype: str
    """
    fl = False
    if num < 0:
        num *= -1
        fl = True
    if num == 0:
        return '0'
    arr = []
    while num > 0:
        n = num % 7
        arr.append(str(n))
        num = (num - n) // 7
    if fl:
        arr.append('-')
    return ''.join(arr[::-1])


def findRelativeRanks(score):
    """
    :type score: List[int]
    :rtype: List[str]
    """

    s = set(score)
    for i in range(len(score)):
        ace = max(s)
        index = score.index(ace)
        if i < 3:
            if i == 0:
                score[index] = "Gold Medal"
            if i == 1:
                score[index] = "Silver Medal"
            if i == 2:
                score[index] = "Bronze Medal"
        else:
            score[index] = str(i + 1)
        s.remove(ace)
    return score


def detectCapitalUse(word):
    """
    :type word: str
    :rtype: bool
    """
    if len(word) == 1:
        return True
    fl1 = word[0].isupper()
    fl2 = word[1].isupper()
    for c in word[1:]:
        if fl1:
            if c.isupper() != fl2:
                return False
        else:
            if c.isupper():
                return False
    return True


def reverseStr(s, k):
    """
    :type s: str
    :type k: int
    :rtype: str
    """
    list1 = list(s)
    end = k
    start = 0
    l = len(list1)
    while end <= l:
        list1[start:end] = list1[start:end][::-1]
        start += 2 * k
        end = start + k
    if start < l < end:
        list1[start:l] = list1[start:l][::-1]
    return ''.join(list1)


def checkRecord(s):
    """
    :type s: str
    :rtype: bool
    """
    return s.count('A') < 2 and s.find('LLL') == -1


def reverseWords(s):
    """
    https://leetcode.com/problems/reverse-words-in-a-string-iii/
    :type s: str
    :rtype: str
    """
    list1 = s.split(' ')
    for i in range(len(list1)):
        list1[i] = list1[i][::-1]
    return ' '.join(list1)


def arrayPairSum(nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    nums.sort()
    sum = 0
    for i in range(0, len(nums), 2):
        sum += nums[i]
    return sum


def findLHS(nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    mydict = {}
    maxl = 0
    set1 = set(nums)
    for num in nums:
        if num not in mydict:
            mydict[num] = 1
        else:
            mydict[num] += 1
    for number in set1:
        if number - 1 in set1:
            temp = mydict[number] + mydict[number - 1]
            maxl = temp if temp > maxl else maxl

    return maxl


def findRestaurant(list1, list2):
    """
    :type list1: List[str]
    :type list2: List[str]
    :rtype: List[str]
    """
    mylist = []
    mydict = {}
    temp = len(list1) + len(list2)
    for i in range(len(list1)):
        mydict[list1[i]] = i
    for j in range(len(list2)):
        common_string = list2[j]
        if common_string in mydict:
            sum_indexes = j + mydict[common_string]
            if sum_indexes == temp:
                mylist.append(common_string)
            elif sum_indexes < temp:
                mylist.clear()
                mylist.append(common_string)
                temp = sum_indexes
            else:
                pass
    return mylist


def canPlaceFlowers(flowerbed, n):
    """
    :type flowerbed: List[int]
    :type n: int
    :rtype: bool
    """
    tmp = -1
    sum = 0
    l = len(flowerbed)
    i = 0
    while i < l + 1:
        if i == l or flowerbed[i]:
            j = i - tmp - 1
            if tmp < 0 or i == l:
                if j == 2:
                    sum += 1
                elif j == 0:
                    pass
                else:
                    if j % 2 == 0:
                        sum += (j) // 2
                    else:
                        sum += (j - 1) // 2
            elif j % 2 == 0:
                sum += (j - 2) // 2
            else:
                sum += (j - 1) // 2
            tmp = i
        i += 1
    return sum == n


def checkPerfectNumber(num):
    """
    :type num: int
    :rtype: bool
    """
    if num == 1:
        return True
    sum = 1
    i = 2
    j = int(num / 2)
    while i < j:
        if num % i == 0:
            j = int(num / i)
            if i != j:
                sum += i + j
            else:
                sum += i
            print(i, j, sum)
        i += 1
    return True if sum == num else False


def busyStudent(startTime, endTime, queryTime):
    """
    :type startTime: List[int]
    :type endTime: List[int]
    :type queryTime: int
    :rtype: int
    """
    count = 0
    for x, y in zip(startTime, endTime):
        if x <= queryTime <= y:
            count += 1
    return count


def getConcatenation(nums):
    """
    :type nums: List[int]
    :rtype: List[int]
    """
    nums.extend(nums)
    return nums


def runningSum(nums):
    """
    :type nums: List[int]
    :rtype: List[int]
    """
    for i in range(1, len(nums)):
        nums[i] = nums[i] + nums[i - 1]
    return nums


def maxIncreaseKeepingSkyline(grid):
    """
    :type grid: List[List[int]]
    :rtype: int
    """
    max_row = []
    max_col = []
    for i in range(len(grid)):
        max_row.append(max(grid[i]))
        m = 0
        for j in range(len(grid)):
            m = grid[j][i] if grid[j][i] > m else m
        max_col.append(m)
    sum = 0
    for p in range(len(grid)):
        for q in range(len(grid)):
            sum += min(max_row[p], max_col[q]) - grid[p][q]
    return sum


def garbageCollection(garbage, travel):
    """
    :type garbage: List[str]
    :type travel: List[int]
    :rtype: int
    """
    mpg = ['M', 'P', 'G']
    time = [0] * 3
    ptr = [0] * 3
    i: int
    for i in range(len(garbage)):
        unit: int
        for unit in range(len(mpg)):
            avl = garbage[i].count(mpg[unit])
            time[unit] += avl
            if avl != 0:
                ptr[unit] = i
    for u in range(len(mpg)):
        for j in range(0, ptr[u]):
            time[u] += travel[j]
    return sum(time)


def restoreString(s, indices):
    """
    :type s: str
    :type indices: List[int]
    :rtype: str
    """
    s1 = [None] * len(s)
    for idx, c in zip(indices, s):
        s1[idx] = c
    return ''.join(s1)


def countMatches(items, ruleKey, ruleValue):
    """
    :type items: List[List[str]]
    :type ruleKey: str
    :type ruleValue: str
    :rtype: int
    """
    if ruleKey == "type":
        kw = 0
    elif ruleKey == "color":
        kw = 1
    else:
        kw = 2
    count = 0
    for item in items:
        if item[kw] == ruleValue:
            count += 1
    return count


def groupThePeople(groupSizes):
    """
    :type groupSizes: List[int]
    :rtype: List[List[int]]
    """
    dict = {}
    ans = []
    for i in range(len(groupSizes)):
        g = groupSizes[i]
        if g in dict:
            dict[g].append(i)
        else:
            dict[g] = [i]

    for num in dict:
        begin = 0
        end = num
        while end <= len(dict[num]):
            ans.append(dict[num][begin:end])
            begin = end
            end += num
    return ans


def minOperations(boxes):
    """
    :type boxes: str
    :rtype: List[int]
    """
    dict = collections.defaultdict(list)
    for i, j in enumerate(boxes):
        dict[int(j)].append(i)
    res = []
    for k in range(len(boxes)):
        res.append(sum([abs(num - k) for num in dict[1]]))
    return res


def countGoodSubstrings(s):
    """
    :type s: str
    :rtype: int
    """
    l = len(s)
    if l < 3:
        return 0
    count = 0
    for p in range(l - 2):
        if len(set(s[p:p + 3])) == 3:
            count += 1
    return count


def numOfSubarrays(arr, k, threshold):
    """
    :type arr: List[int]
    :type k: int
    :type threshold: int
    :rtype: int
    """
    if len(arr) < k:
        return 0
    start, end = 0, k
    count = 0
    sum1 = sum(arr[start:end])
    if sum1 // k >= threshold:
        count += 1
    while end < len(arr):
        sum1 += arr[end] - arr[start]
        if sum1 // k >= threshold:
            count += 1
        start += 1
        end += 1
    return count


def findMaxSequenceOnes(nums):
    count = 0
    ans = 0
    for i in range(len(nums)):
        if nums[i] == 1:
            count += 1
        else:
            ans = max(ans, count)
            count = 0
    ans = max(ans, count)
    return ans


def findMaxConsecutiveOnes(nums):
    """
    Given a binary array, find the maximum number of consecutive 1s in this array if you can flip at most one 0.
    :param nums:
    :return:
    """
    # [0,0,1,1,0,0,1,1,1,0,1,1,0,0,0,1,1,1,1]
    count0, count1 = 0, 0
    ans = 0
    k = 1
    i = 0
    res = 0
    while i < len(nums):
        if nums[i] == 1:
            if count0 == 1 and k != 0:
                count1 += 1
                count0 = 0
                k -= 1
                res = i
            elif count0 > 0:
                ans = max(ans, count1 + k)
                if res:
                    i = res
                    res = 0
                count1 = 0
                count0 = 0
                k = 1
            count1 += 1
        else:
            count0 += 1
        i += 1
    ans = max(ans, count1 + k)
    return ans


def longestOnes(nums, k):
    """
    :type nums: List[int]
    :type k: int
    :rtype: int
    """
    # [1,1,1,0,0,0,1,1,1,1,0] k=2
    # left pointer for tracking used k
    left = 0
    for i in range(len(nums)):  # i pointer always move up
        if nums[i] == 0:
            k -= 1  # reduce k if 0
        if k < 0:  # when run out of k, back window slices up with front window
            if nums[left] == 0:
                k += 1  # return value of k if back window is 0
            left += 1
    return i - left + 1  # return largest size of window


def longestOnes1(A, K):
    # [0,0,1,1,0,0,1,1,1,0,1,1,0,0,0,1,1,1,1]
    i = 0
    for j in range(len(A)):
        K -= 1 - A[j]
        if K < 0:
            K += 1 - A[i]
            i += 1
    return j - i + 1


def numberOfSubstrings(s):
    """
    :type s: str
    :rtype: int
    """
    # 'aabaacabc'
    # ans = 0
    # left, right = 0, 0
    # l = len(s)
    # while left < l-2:
    #     for i in range(left+2, l):
    #         s1 = s[left:i+1]
    #         if 'a' in s1 and 'b' in s1 and 'c' in s1:
    #             ans += l - i
    #             break
    #     left += 1
    # return ans
    ans = 0
    dict = {'a': 0, 'b': 0, 'c': 0}
    j = 0
    for i in range(len(s)):
        c = s[i]
        dict[c] += 1
        while j < len(s) and dict['a'] > 0 and dict['b'] > 0 and dict['c'] > 0:
            dict[s[j]] -= 1
            j += 1
        print(j)
        ans += j
    return ans


def divisorSubstrings(num, k):
    """
    :type num: int
    :type k: int
    :rtype: int
    """
    ans = 0
    s = str(num)
    for i in range(k, len(s) + 1):
        n = int(s[i - k:i])

        if n != 0 and num % n == 0:
            ans += 1
    return ans


def minimumRecolors(blocks, k):
    """
    :type blocks: str
    :type k: int
    :rtype: int
    """
    res = len(blocks)
    count = 0
    for i in range(res):
        if blocks[i] == 'W':
            count += 1
        if i > k - 2:
            if count < res:
                res = count
            if blocks[i - k + 1] == 'W':
                count -= 1
    return res


def minimumDifference(nums, k):
    """
    :type nums: List[int]
    :type k: int
    :rtype: int
    """
    l = len(nums)
    if k > l:
        return 0
    nums.sort()
    res = nums[-1] - nums[0]
    for i in range(k - 1, l):
        n = nums[i] - nums[i - k + 1]
        if n < res:
            res = n
    return res


def maxSumTwoNoOverlap(nums, firstLen, secondLen):
    """
    :type nums: List[int]
    :type firstLen: int
    :type secondLen: int
    :rtype: int
    """
    def findMax(nums,k):
        sm = 0
        l, r = 0, 0
        if len(nums) < k:
            return 0, l, r
        for j in range(k):
            sm += nums[j]
        mx = sm
        r = j
        for i in range(k,len(nums)):
            sm += nums[i]
            sm -= nums[i-k]
            if sm > mx:
                mx = sm
                r = i
                l = i-k+1
        return mx, l, r
    # Find firstlen
    maxFirst, l, r = findMax(nums, firstLen)
    maxSecond1,g,h = findMax(nums[0:l], secondLen)
    maxSecond2,g,h = findMax(nums[r+1:], secondLen)
    print(maxFirst, maxSecond1, maxSecond2, l, r)
    res1 = max(maxSecond1, maxSecond2) + maxFirst
    #Find secondlen
    maxSecond, l, r = findMax(nums, secondLen)
    maxFirst1,g,h = findMax(nums[0:l], firstLen)
    maxFirst2,g,h = findMax(nums[r+1:], firstLen)
    print(maxSecond, maxFirst1, maxFirst2, l, r)
    res2 = max(maxFirst1, maxFirst2) + maxSecond
    return max(res1,res2)


def maxSumTwoNoOverlap(nums, firstLen, secondLen):
    """
    :type nums: List[int]
    :type firstLen: int
    :type secondLen: int
    :rtype: int
    """

    def findMax(arr, L, M):
        """
        Support function to find the max sum of 2 subarray with given lengths
        :param arr:
        :param L:
        :param M:
        :return:
        """
        sumL, sumM = 0, 0
        # Find sum of subarray L and M at the beginning. |____L____|____M_____|------------------
        for i in range(L + M):
            if i < L:
                sumL += arr[i]
            else:
                sumM += arr[i]

        maxL = sumL
        ans = sumL + sumM
        # Continuously find the max of subarray length L and value of subarray length M. The answer is the maximum
        # combination.
        for j in range(L + M, len(arr)):
            sumL += arr[j - M] - arr[j - M - L]
            maxL = max(maxL, sumL)  # find max of sub-L
            sumM += arr[j] - arr[j - M]  # find sub-M
            ans = max(ans, maxL + sumM)  # answer is the max combination
        return ans  # -------|max sub-L|--------|______M_______|

    # firstLen can happen before or after secondLen
    return max(findMax(nums, firstLen, secondLen), findMax(nums, secondLen, firstLen))


def numberOfSubarrays(nums, k):
    """
    Sliding window with elastic ends
    :type nums: List[int]
    :type k: int
    :rtype: int
    """
    l, ans, suml = 0, 0, 0
    for i in range(len(nums)):  # keep sliding forward
        if nums[i] % 2 == 1:
            k -= 1
        if k < 1:
            suml = 0
        while k < 1:  # reduce left and count # of diff
            suml += 1
            if nums[l] % 2 == 1:
                k += 1
            l += 1
        ans += suml  # adding up # of diff from left
    return ans


def maxConsecutiveAnswers(answerKey, k):
    """
    Sliding window to find max size with allowing k changes
    :type answerKey: str
    :type k: int
    :rtype: int
    """

    def helper(arr, k, c):
        l = 0
        i = 0
        for i in range(len(arr)):
            if arr[i] != c:
                k -= 1
            if k < 0:
                if arr[l] != c:
                    k += 1
                l += 1
        return i - l + 1

    return max(helper(answerKey, k, 'T'), helper(answerKey, k, 'F'))

def maxDepth(root):
    if not root:
        return 0
    q = collections.deque()
    q.append((root, 1))
    while len(q) > 0:
        node, ans = q.popleft()
        if node.left:
            q.append((node.left, ans+1))
        if node.right:
            q.append((node.right, ans+1))
    return ans


def findTarget(root, k):
    """
    :type root: TreeNode
    :type k: int
    :rtype: bool
    """
    if not root.left and not root.right:
        return False
    dic = {}
    queue = []
    queue.append(root)
    while len(queue) > 0:
        node = queue.pop(0)
        target = k - node.val
        if target in dic:
            return True
        dic[node.val] = 0
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    return False

def deepestLeavesSum(root):
    """
    :type root: TreeNode
    :rtype: int
    """
    if not root.left and not root.right:
        return root.val
    q = []
    sm = 0
    q.append(root)
    while len(q) > 0:
        temp = []
        while len(q) > 0:
            node = q.pop()
            sm += node.val
            if node.left:
                temp.append(node.left)
            if node.right:
                temp.append(node.right)
        q = temp
    return sm

def maxVowels(s, k):
    """
    Sliding window size k, find max # of interest elements in that window
    :type s: str
    :type k: int
    :rtype: int
    """
    set1 = {'a', 'o', 'e', 'u', 'i'}
    l = len(s)
    if l < k:
        return 0
    sum1 = 0
    ans = 0
    for i in range(l):
        if s[i] in set1:
            sum1 += 1
        if i >= k - 1:
            ans = max(ans, sum1)
            if s[i - k + 1] in set1:
                sum1 -= 1
    return ans


def threeSum(nums):
    """
    :type nums: List[int]
    :rtype: List[List[int]]
    """

    def findTwo(nums, k):
        dic = {}
        arr = []
        for i in range(k + 1, len(nums)):
            # if i == k:
            #     continue
            target = -nums[k] - nums[i]
            if target in dic:
                arr.append(sorted([nums[k], target, nums[i]]))
            dic[nums[i]] = i
        return arr

    a = []
    for j in range(len(nums)):
        arr = findTwo(nums, j)
        if len(arr) > 0:
            print(arr)
            for l in arr:
                if l not in a:
                    a.append(l)
    return a


def getTargetCopy(original, cloned, target):
    """
    BFS to find corresponding node on the cloned tree from a given node on original tree
    :type original: TreeNode
    :type cloned: TreeNode
    :type target: TreeNode
    :rtype: TreeNode
    """
    q1 = collections.deque()
    q2 = collections.deque()

    q1.append(original)
    q2.append(cloned)
    while True:
        node1 = q1.popleft()
        node2 = q2.popleft()
        if node1 == target:
            return node2
        if node1.left:
            q1.append(node1.left)
            q2.append(node2.left)
        if node1.right:
            q1.append(node1.right)
            q2.append(node2.right)


def mergeTrees(root1, root2):
    """
    :type root1: TreeNode
    :type root2: TreeNode
    :rtype: TreeNode
    """
    q = collections.deque()
    q.append(root1)
    q.append(root2)
    while len(q) != 0:
        node1 = q.popleft()
        node2 = q.popleft()
        if node1 and node2:
            node1.val = node1.val + node2.val
            if not node1.left and node2.left:
                node1.left = TreeNode(0)
            q.append(node1.left)
            q.append(node2.left)
            if not node1.right and node2.right:
                node1.right = TreeNode(0)
            q.append(node1.right)
            q.append(node2.right)
        elif node1:
            q.append(node1.left)
            q.append(None)
            q.append(node1.right)
            q.append(None)

    return root1


def sumEvenGrandparent(root):
    """
    :type root: TreeNode
    :rtype: int
    """
    def sumG(node):
        res = 0
        if node.left and node.left.left:
            res += node.left.left.val
        if node.left and node.left.right:
            res += node.left.right.val
        if node.right and node.right.left:
            res += node.right.left.val
        if node.right and node.right.right:
            res += node.right.right.val
        return res
    if not root:
        return 0
    sm = 0
    if root.val % 2 == 0:
        sm = sumG(root)
    return sm + sumEvenGrandparent(root.left) + sumEvenGrandparent(root.right)


def reverseOddLevels(root):
    """
    :type root: Optional[TreeNode]
    :rtype: Optional[TreeNode]
    """
    q = [root]
    lv = 0
    while len(q) > 0:
        if lv % 2 == 1:
            for i in range(2**lv // 2):
                q[i].val, q[2**lv-i-1].val = q[2**lv-i-1].val, q[i].val
        for j in range(len(q)):
            node = q.pop(0)
            q.append(node.left)
            q.append(node.right)
        lv += 1
    return root


def maximumUniqueSubarray(nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    s_arr = [] + nums
    for i in range(1, len(s_arr)):
        s_arr[i] += s_arr[i-1]

    left = 0
    ans = nums[left]
    set1 = set()
    for j in range(len(nums)):
        n = nums[j]
        if n in set1:
            if left == 0:
                ans = max(ans, s_arr[j-1])
            else:
                ans = max(ans, s_arr[j-1] - s_arr[left-1])
            while n in set1:
                #print(set1,nums[left])
                set1.remove(nums[left])
                left += 1
        set1.add(n)
    if left == 0:
        ans = max(ans, s_arr[j])
    else:
        ans = max(ans, s_arr[j] - s_arr[left - 1])
    return ans


# 967
def numsSameConsecDiff(n, k):
    """
    :type n: int
    :type k: int
    :rtype: List[int]
    """
    ans = []
    for i in range(1, 10):
        q = [i]
        for a in range(n-1):
            for j in range(len(q)):
                num = q.pop(0)
                tmp = num % 10
                if tmp + k < 10:
                    q.append(num * 10 + tmp + k)
                if tmp - k >= 0 and k != 0:
                    q.append(num * 10 + tmp - k)
        ans.extend(q)
    return ans

# 107
def levelOrderBottom(root):
    """
    :type root: TreeNode
    :rtype: List[List[int]]
    """
    if not root:
        return []
    q = [root]
    ans = []
    while len(q) > 0:
        tmp = []
        for _ in range(len(q)):
            node = q.pop(0)
            tmp.append(node.val)
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        ans.append(tmp)
    return ans[::-1]


# 1448
def goodNodes(root):
    """
    :type root: TreeNode
    :rtype: int
    """
    def count(prev, node):
        if not node:
            return 0
        mx = max(prev, node.val)
        if prev <= node.val:
            return 1 + count(mx, node.left) + count(mx, node.right)
        else:
            return count(mx, node.left) + count(mx, node.right)
    return count(-10**4, root)


# 930
def numSubarraysWithSum(nums, goal):
    """
    :type nums: List[int]
    :type goal: int
    :rtype: int
    """
    count = 0
    ans = 0
    left = 0
    for i in range(len(nums)):
        if nums[i] == 1:
            goal -= 1
        if goal < 1:
            count = 0
            while goal < 1:
                if nums[left] == 1:
                    goal += 1
                left += 1
                count += 1
        ans += count
    return ans


# 1052
def maxSatisfied(customers, grumpy, minutes):
    """
    :type customers: List[int]
    :type grumpy: List[int]
    :type minutes: int
    :rtype: int
    """
    left = 0
    notG_customers = 0
    G_customers = 0
    max_c = 0
    for i in range(len(customers)):
        minutes -= 1
        if grumpy[i] == 0:
            notG_customers += customers[i]

        if grumpy[i] == 1:
            G_customers += customers[i]

        if minutes < 0:
            if grumpy[left] == 1:
                G_customers -= customers[left]
            left += 1
        max_c = max(max_c, G_customers)

    return max_c + notG_customers


print(maxSatisfied([1],[0],1))
