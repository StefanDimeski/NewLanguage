from newlang.ast.node import *
from newlang.flag import *
from newlang.utils import *


class While_Stat(Node):
    def __init__(self, cond, internal_block):
        super().__init__()
        self.cond = cond
        self.internal_block = internal_block

    def exec(self, ar):
        while self.cond.exec(ar) == True:
            ret = self.internal_block.exec(ar)

            if ret is not None:
                first_item = ret.pop(0)

                if first_item[0] == Flag.BREAK:
                    if len(ret) > 0:
                        return ret
                    else:
                        return None

        return None

    def compile(self, ar, end_function_lbl, loop_continue_lbls=None, loop_break_lbls=None):
        if loop_break_lbls is None:
            loop_break_lbls = []
        if loop_continue_lbls is None:
            loop_continue_lbls = []


        mem = ar.mem

        begin_lbl = mem.new_label()
        end_lbl = mem.new_label()

        loop_continue_lbls.append(begin_lbl)
        loop_break_lbls.append(end_lbl)

        printc(f"{begin_lbl}:")
        res_varname = self.cond.compile(ar)
        ar.mem.make_non_evictible(res_varname)

        reg = mem.reg_name(mem.get_var_in_any_reg(res_varname))

        printc(f"beqz {reg}, {end_lbl}")
        self.internal_block.compile(ar, end_function_lbl=end_function_lbl, loop_continue_lbls=loop_continue_lbls, loop_break_lbls=loop_break_lbls)
        printc(f"j {begin_lbl}")
        printc(f"{end_lbl}:")

        if mem.is_var_temp(res_varname):
            mem.delete_var(res_varname)