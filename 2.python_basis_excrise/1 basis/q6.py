'''
斐波那契数列（Fibonacci sequence），又称黄金分割数列，指的是这样一个数列：0、1、1、2、3、5、8、13、21、34、……。

'''

import math
import matplotlib
import numpy as np

a =[]
a.append(0)
a.append(1)

Num = int(input("please input some numbers!"))

for i in range(2,Num,1):
    res = a[i-1]+a[i-2]
    a.append(res)
    
for j in a:
    print(j)

print(a)