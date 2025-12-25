#请完善如下代码
"""
43、获得用户输入的以逗号分隔的三个数字，记为a、b、c，以a为起始数值，b为差，c为数值的数量，
产生一个递增的等差数列，将这个数列以列表格式输出，请完善PY103.py中代码。
"""
a, b, c = eval(input())
ls = []
for i in range(c):
    ls.append(a+b*i)
print(ls)

