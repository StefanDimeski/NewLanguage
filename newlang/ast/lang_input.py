from newlang.ast.node import *
from newlang.utils import *

class Input(Node):
    def __init__(self):
        super().__init__()
        pass

    def exec(self, ar):
        return int(input())

    def compile(self, ar):
        mem = ar.mem

        printc("li $v0, 5")
        printc("syscall")

        varname = mem.get_rand_varname()
        free_reg = mem.reg_name(mem.init_new_var(varname, temp=True))

        printc(f"move {free_reg}, $v0")

        return varname