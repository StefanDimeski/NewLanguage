from newlang.ast.node import *
from newlang.utils import *

class If_Statement(Node):
    def __init__(self, cond, block, else_block):
        super().__init__()
        self.condition = cond
        self.block = block
        self.else_block = else_block

    def exec(self, ar):
        condition = self.condition.exec(ar)
        if not isinstance(condition, bool):
            return None

        if condition:
            ret = self.block.exec(ar)
            if ret is not None:
                return ret
        else:
            if self.else_block != None:
                ret = self.else_block.exec(ar)
                if ret is not None:
                    return ret

        return None

    def compile(self, ar, end_function_lbl=None, loop_continue_lbls=None, loop_break_lbls=None):
        if loop_break_lbls is None:
            loop_break_lbls = []
        if loop_continue_lbls is None:
            loop_continue_lbls = []

        mem = ar.mem

        result_varname = self.condition.compile(ar)
        result_reg = mem.reg_name(mem.get_var_in_any_reg(result_varname))

        else_label = mem.new_label()
        printc(f"beqz {result_reg}, {else_label}")

        if mem.is_var_temp(result_varname):
            mem.delete_var(result_varname)

        self.block.compile(ar, end_function_lbl=end_function_lbl, loop_continue_lbls=loop_continue_lbls, loop_break_lbls=loop_break_lbls)

        if self.else_block == None:
            printc(f"{else_label}:")
        else:
            after_if_label = mem.new_label()
            printc(f"j {after_if_label}")
            printc(f"{else_label}:")
            self.else_block.compile(ar, end_function_lbl=end_function_lbl, loop_continue_lbls=loop_continue_lbls, loop_break_lbls=loop_break_lbls)
            printc(f"{after_if_label}:")