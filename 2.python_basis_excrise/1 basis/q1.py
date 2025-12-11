'''
题目：有四个数字：1、2、3、4，能组成多少个互不相同且无重复数字的三位数？各是多少？

'''

arr=[]
res = 0
for i in range(1,5,1):
    for j in range(1,5,1):
        for k in range(1,5,1):
                sum = 100*i+10*j+k
                if (i!=j and i!=k and j!=k):
                    arr.append(sum)

for i in arr:
     print(i)   
print(len(arr))