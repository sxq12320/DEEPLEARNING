'''
将一个正整数分解质因数。例如：输入90,打印出90=2*3*3*5。

'''
global T_judge 
T_judge = 0

def jie(a):
    for i in range(2,a+1,1):
        if(a%i == 0):
            return i
        return 0

Num = int(input("please input a number!"))




res = []
for i in range(0,100,1):
    if (jie(Num) == 1):
        res.append()
