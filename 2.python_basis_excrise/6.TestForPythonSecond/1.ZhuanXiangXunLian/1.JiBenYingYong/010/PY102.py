#
# 在____________上补充代码
#

"""
42、在考生文件夹下有个文件PY102.py，在横线处填写代码，完成如下功能。
让用户输入一串数字和字母混合的数据，然后统计其中数字和字母的个数，显示在屏幕上。
例如：输入：Fda243fdw3输出：数字个数:4,字母个数:6
"""


ns = input("请输入一串数据：")
dnum,dchr = 0,0
for i in ns:
    if i.isnumeric():
        dnum += 1
    elif i.isalpha():
        dchr += 1
    else:
        pass
print('数字个数：{}，字母个数：{}'.format(dnum , dchr))
