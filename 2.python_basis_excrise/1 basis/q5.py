'''
输入三个整数x,y,z，请把这三个数由小到大输出

'''

class Number:
     
    @staticmethod
    def max(a,b,c):
        if(a>=b and a>=c):
            return a
        elif(b>=a and b>=c):
            return b
        elif(c>=a and c>=b):
            return c

    @staticmethod
    def min(a,b,c):
        if (a<=b and a<=c):
            return a
        elif(b<=a and b<=c):
            return b
        elif(c<=a and c<=b):
            return c
        
X=int(input("x?"))
Y=int(input("Y?"))
Z=int(input("Z?"))

max = Number.max(X,Y,Z)
min = Number.min(X,Y,Z)

res = []
res.append(min)
res.append(X+Y+Z-max-min)
res.append(max)

for i in res:
    print(i)
