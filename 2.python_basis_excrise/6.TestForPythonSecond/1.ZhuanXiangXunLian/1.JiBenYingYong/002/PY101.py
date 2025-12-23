# 请在______处使用一行代码替换
#
# 注意：请不要修改其他已给出代码

"""
41、考生文件夹下存在一个文件PY101.py，请写代码替换横线，不修改其他代码，实现以下功能:
随机选择一一个手机品牌屏幕输出。
"""
import random
brandlist = ['华为','苹果','诺基亚','OPPO','小米']
random.seed(0)
name = brandlist[random.randint(0,4)]
print(name)
