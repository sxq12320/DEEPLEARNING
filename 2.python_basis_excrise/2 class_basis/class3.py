'''
算数运算符符号
'''
import math

num1 = 10
num2 = 20
num3 = 40
num4 = 50

add = num1+num2
cut = num4-num3
cheng = num3*num2
chu = num4/num1
zhenfgchu = num4//num1

print(type(chu)) #c除法的商一定是浮点型
print(chu)

print(zhenfgchu)
print(type(zhenfgchu))#这个是整除哈不管四舍五入的原则，直接忽略小数，且他的商必定是整形

mizhishu = num1**3
print(mizhishu)

name = input("请输入你的年龄：")
print(name,type(name))
name = int(name)
print(type(name))

print("年龄\n生日\n姓名\n")
print("年龄\t生日\t姓名\t")
print("年龄\r生日\r姓名\r")