import math


def toHex(num):
    """
    :type num: int
    :rtype: str
    """
    s = "abcdef"
    if num == 0:
        return '0'
    if num < 0:
        num += 2**32
    res = []
    while num:
        d = num%16
        res.append(str(d) if d < 9 else s[d-10])
        num //= 16
    return ''.join(res[::-1]) #if num else '0'


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
        if num%2 == 0:
            length += num
        else:
            f =True
            if num > 1:
                length += num-1

    return length+1 if f else length


def addStrings(num1, num2):
    """
    :type num1: str
    :type num2: str
    :rtype: str
    """
    n = len(num1)
    m = len(num2)
    l=0
    sum = 0
    while(n > 0 and m > 0):
        sum += (int(num1[n-1]) + int(num2[m-1]))*10**l
        l += 1
        n -= 1
        m -= 1

    if n == 0 and m != 0:
        sum += int(num2[:m])*10**l
    elif m == 0 and n != 0:
        sum += int(num1[:n])*10**l
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
        if (i+1) not in myset:
            mylist.append(i+1)

    return mylist


def findContentChildren(g, s):
    """
    :type g: List[int]
    :type s: List[int]
    :rtype: int
    """
    g.sort()
    s.sort()
    i = len(g)-1
    j = len(s)-1
    count = 0
    while i>=0 and j>=0:
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
        if area%mid == 0:
            return [int(area/mid),mid]
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
        end = t0 + duration -1
        if i < l-1:
            t1 = timeSeries[i+1]
            if end >= t1:
                total += t1-t0
            else:
                total += end-t0+1
        else:
            total += end-t0+1
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
        for j in range(start,l2):
            if nums2[j] > x:
                arr.append(nums2[j])
                break
        if j == l2-1 and nums2[l2-1] <= x:
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
        start += 2*k
        end = start+k
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
    for i in range(0,len(nums),2):
        sum += nums[i]
    return sum


print(arrayPairSum([1,4,3,2]))
