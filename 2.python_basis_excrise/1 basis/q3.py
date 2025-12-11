'''
题目：一个整数，它加上100后是一个完全平方数，再加上168又是一个完全平方数，请问该数是多少？
'''

import math

res = []
for i in range(-100,10000,1):
    test1 = i+100
    test2 = i+100+168
    if(math.sqrt(test1)==int(math.sqrt(test1)) and math.sqrt(test2)==int(math.sqrt(test2)) ):
        res.append(i)
    else:
        continue

for j in res:
    print(j)

print("\n")
print(len(res))
