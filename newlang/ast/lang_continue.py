from newlang.ast.node import *
from newlang.utils import *
from newlang.flag import *

class Continue(Node):
    def __init__(self):
        super().__init__()

    def exec(self, ar):
        return [(Flag.CONTINUE,)]

    def compile(self, ar, loop_continue_lbls=None, loop_break_lbls=None):
        if loop_break_lbls is None:
            loop_break_lbls = []
        if loop_continue_lbls is None:
            loop_continue_lbls = []
        assert len(loop_continue_lbls) == len(loop_break_lbls)

        if len(loop_continue_lbls) == 0:
            print("ERROR: 'continue' outside of loop")
            exit()

        printc(f"j {loop_continue_lbls[-1]}")