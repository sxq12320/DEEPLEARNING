'''
输入某年某月某日，判断这一天是这一年的第几天？

'''

import math


def Judge_runnian(number):
    if(number%4==0 and number%400!=0):
        return 1
    else:
        return 0
    
Y1 = [0,31, 29, 31 ,30, 31, 30, 31, 31, 30, 31, 30, 31]
Y2 = [0,31, 28 ,31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

Year = int(input("请输入年份："))
Month = int(input("month?"))
Day = int(input("day?"))
SumDay = 0

judge = Judge_runnian(Year)

if (judge == 1):
    for i in range(1,Month,1):
        SumDay = SumDay+Y1[i]

elif(judge == 0):
    for i in range(1,Month,1):
        SumDay = SumDay+Y2[i]

SumDay = SumDay+Day
print(SumDay)
