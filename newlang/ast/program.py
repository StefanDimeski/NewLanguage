from newlang.ast.node import *
from newlang.utils import *
from newlang.flag import *
from newlang.ar import *

class Program(Node):
    def __init__(self, funcs, block):
        super().__init__()
        self.funcs = funcs
        self.block = block

    def exec(self, ar):
        ret = self.block.exec(ar)

        if ret is not None:
            first_item = ret.pop(0)
            if first_item[0] == Flag.RETURN:
                print(f"ERROR: 'return' outside of function body")
            elif first_item[0] == Flag.CONTINUE:
                print(f"ERROR: 'continue' outside of loop")
            elif first_item[0] == Flag.BREAK:
                print(f"ERROR: 'break' outside of loop")

    def compile(self):
        printc(f"j main_program")
        printc("")
        printc("")
        printc("")

        for func in self.funcs:
            func.compile(None)
            printc("")

        printc("")

        printc(f"main_program:")
        ar = AR()
        ar.init()
        self.block.compile(ar)

        ar = ar.destroy()

        # end program
        printc("li $v0, 10")
        printc("syscall")