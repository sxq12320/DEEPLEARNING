'''
输出 9*9 乘法口诀表。

'''

for i in range(1,10,1):
    print()
    for j in range(1,i+1,1):
        res = i*j
        print("%dx%d=%d "%(j,i,res),end="")
