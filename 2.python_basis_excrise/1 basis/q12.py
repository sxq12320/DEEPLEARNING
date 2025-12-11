'''
判断101-200之间有多少个素数，并输出所有素数。
'''

import math
def judge(a):
    t = 1
    for i in range(2,int(math.sqrt(a))+1,1):
        if(a%i == 0):
            t = 0
    return t


res = 0
for i in range(101,201,1):
    if(judge(i)==1):
        print(i)
        res+=1
print(res)