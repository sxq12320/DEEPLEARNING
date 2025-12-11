class Sxq:
    name = "shixiaoqiang"
    age = 20

    @staticmethod###这个静态方法一定要学会使用
    def add(a,b):
        return (a+b)
       
d= 10
f= 100
sum = Sxq.add(d,f)
print(sum)