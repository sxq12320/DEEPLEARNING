from manim import *

class CreateCircle (Scene):
    def construct(self) -> None:
        circle = Circle()
        circle.set_fill(PINK , opacity=0.5)
        self.play(Create(circle))
        self.wait(1)
