from manim import *

class Chapter_1(Scene):
    def __init__(self):
        super(Chapter_1 , self).__init__()
        self.text1 = Text("你好陌生人，这是我的深度学习入门教程" , font_size=50).move_to((ORIGIN))
        self.text2 = Text("现在我们就从最基本的全连接神经网络讲起" , font_size=50).move_to((ORIGIN))
    def construct(self):

