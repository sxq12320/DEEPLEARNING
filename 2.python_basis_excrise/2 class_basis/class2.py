'''
可以采用 type函数进行检测某个变量的数据类型

整形  int

浮点型 float

bool型   这个是有固定的写法的哈
一个True;一个为False
布尔值可以当作整形进行对待
True相当于1 False 相当于0

complex复数形
z=a+bj
虚数单位只能是j

字符型 str
需要加上引号，单双银行均可，包含多行内容也可以使用单引号
'''

num1 =5
print(type(num1))

age = 18
UserNmae = 'shixiaoqiiang'
print("我的名字：%s , 我的年龄：%d"%(UserNmae,age))

num2 = 2.345789
print("%.2f"%num2)

print("%08d"%num1)

print(f"我的名字是{UserNmae},我今年{age}岁了")