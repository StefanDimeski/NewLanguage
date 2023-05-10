from newlang.ast.node import *
from newlang.flag import *
from newlang.utils import *

class Return(Node):
    def __init__(self, to_ret):
        super().__init__()
        self.to_return = to_ret

    def exec(self, ar):
        return [(Flag.RETURN, self.to_return.exec(ar))]

    def compile(self, ar, end_function_lbl):
        res_varname = self.to_return.compile(ar)
        res_idx = ar.mem.get_var_in_any_reg(res_varname)

        printc(f"move $v0, {ar.mem.reg_name(res_idx)}")
        printc(f"j {end_function_lbl}")