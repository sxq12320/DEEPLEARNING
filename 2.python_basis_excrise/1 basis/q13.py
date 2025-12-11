'''
水仙花数

'''

def multiply(n):
    res = n*n*n
    return res


def judge(a):
    t = 0
    a_g = a%10
    a_s = (a//10)%10
    a_b = a//100
    sum = multiply(a_g)+multiply(a_b)+multiply(a_s)
    if(sum==a):t = 1
    return t

res = 0
for i in range(100,1000,1):
    if (judge(i)==1):
        res = res+1
        print(i)

print(res)