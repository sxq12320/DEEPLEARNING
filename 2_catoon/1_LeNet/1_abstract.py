from manim import *

class SimpleCircle(Scene):
    def construct(self):
        TEXT = Text("helllo world")
        CIRCLE = Circle(radius=3 , color=WHITE , fill_opacity=0.5)
        self.play(Write(TEXT) , time = 0.5)
        self.wait(2)
        self.play(FadeOut(TEXT))
        self.wait(2)
        self.play(Create(CIRCLE))
        self.wait(3)

        
