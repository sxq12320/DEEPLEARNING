'''
利用条件运算符的嵌套来完成此题：学习成绩>=90分的同学用A表示，60-89分之间的用B表示，60分以下的用C表示。

'''
import math
import matplotlib
import numpy

Score = float(input("please input your score?"))

if (Score>=90):
    print("A")
elif(Score>=60 and Score<=89):
    print("B")
elif(Score<60):
    print("C")
