'''
题目：企业发放的奖金根据利润提成。利润(I)低于或等于10万元时，
奖金可提10%；利润高于10万元，低于20万元时，低于10万元的部分按10%提成，
高于10万元的部分，可提成7.5%；20万到40万之间时，高于20万元的部分，可提成5%；
40万到60万之间时高于40万元的部分，可提成3%；60万到100万之间时，高于60万元的部分，
可提成1.5%，高于100万元时，超过100万元的部分按1%提成，从键盘输入当月利润I，求应发放奖金总数？
'''

import math

class Calculation:

    @staticmethod
    def situation_1(a):
        res1 = a*0.1
        return res1
    
    @staticmethod
    def situation_2(b):
        res2=10*0.1+(b-10)*0.75
        return res2
    
    @staticmethod
    def situation_3(c):
        res3 = 10*0.1+10* 0.75+(c-20)*0.05
        return res3
    
    @staticmethod
    def situation_4(d):
        res4 =10*0.1+10*0.75+20*0.05+(d-40)*0.03
        return res4
    
    @staticmethod
    def situation_5(e):
        res5 = 10*0.1+10*0.75+20*0.05+20*0.03+(e-60)*0.015
        return res5
    
    @staticmethod
    def situation_6(f):
        res6 = 10*0.1+10*0.75+20*0.05+20*0.03+40*0.015+(f-100)*0.01

########################################################################################################################

Number = int(input("请输入今年的利润："))
res = 0

if(Number<=10):
    res = Calculation.situation_1(Number)

elif(Number>10 and Number<=20):
    res = Calculation.situation_2(Number)

elif(Number>20 and Number<=40):
    res = Calculation.situation_3(Number)

elif(Number>40 and Number<=60):
    res = Calculation.situation_4(Number)

elif(Number>60 and Number<=100):
    res = Calculation.situation_5(Number)

elif(Number>100):
    res = Calculation.situation_6(Number)       

print(res)

