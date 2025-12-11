'''
古典问题：有一对兔子，从出生后第3个月起每个月都生一对兔子，
小兔子长到第三个月后每个月又生一对兔子，假如兔子都不死，问每个月的兔子总数为多少？

'''

import numpy

a = []
a.append(1)
a.append(1)

Num = int(input("please input the number for rabbit!"))

for i in range(2,Num,1):
    res = a[i-1]+a[i-2]
    a.append(res)

for j in a:
    print(j)

print(a[Num-1])