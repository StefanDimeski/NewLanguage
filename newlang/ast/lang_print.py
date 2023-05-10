from newlang.ast.node import *
from newlang.utils import *

class Print(Node):
    def __init__(self, expr):
        super().__init__()
        self.expr = expr

    def exec(self, ar):
        res = self.expr.exec(ar)

        print(res)
        #if isinstance(res, str):
         #   print(res)
        #else:
         #   print("ERROR: ARGUMENT TO PRINT MUST BE A STRING!")

        return None

    def compile(self, ar):
        mem = ar.mem

        res_varname = self.expr.compile(ar)
        reg = mem.reg_name(mem.get_var_in_any_reg(res_varname))

        printc(f"li $v0, 1")
        printc(f"move $a0, {reg}")
        printc("syscall")

        if mem.is_var_temp(res_varname):
            mem.delete_var(res_varname)