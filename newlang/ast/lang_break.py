from newlang.ast.node import *
from newlang.ttype import *
from newlang.flag import *
from newlang.utils import *

class Break(Node):
    def __init__(self, lst):
        super().__init__()
        self.lst_to_ret = []
        self.num_breaks = 0
        self.continue_present = False

        for item in lst:
            if item.type == TType.BREAK:
                self.lst_to_ret.append((Flag.BREAK,))
                self.num_breaks += 1
            elif item.type == TType.CONTINUE:
                self.lst_to_ret.append((Flag.CONTINUE, ))
                self.continue_present = True
            else:
                print(f"ERROR: Invalid token in 'break' statement: '{item.val}'")

    def exec(self, ar):
        return self.lst_to_ret.copy()

    def compile(self, ar, loop_continue_lbls=None, loop_break_lbls=None):
        if loop_break_lbls is None:
            loop_break_lbls = []
        if loop_continue_lbls is None:
            loop_continue_lbls = []
        assert len(loop_continue_lbls) == len(loop_break_lbls)

        if len(loop_break_lbls) == 0:
            print("ERROR: 'break' outside of loop")

        if self.continue_present:
            if self.num_breaks + 1 > len(loop_continue_lbls):
                print(f"ERROR: Using {self.num_breaks} 'break' statements and one 'continue' when depth of loop nesting is {len(loop_continue_lbls)}")
                exit()

            printc(f"j {loop_continue_lbls[-self.num_breaks - 1]}")
        else:
            if self.num_breaks > len(loop_continue_lbls):
                print(f"ERROR: Using {self.num_breaks} 'break' statements when depth of loop nesting is {len(loop_continue_lbls)}")
                exit()

            printc(f"j {loop_break_lbls[-self.num_breaks]}")