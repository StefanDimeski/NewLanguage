from newlang.ast.node import *
from newlang.utils import *

class Number(Node):
    def __init__(self, num):
        super().__init__()
        self.num = int(num)

    def exec(self, ar):
        return self.num

    def compile(self, ar):
        mem = ar.mem

        varname = mem.get_rand_varname()
        reg = mem.reg_name(mem.init_new_var(varname, temp=True))
        printc(f"li {reg}, {self.num}")
        return varname