from newlang.ast.node import *
from newlang.ast.assignment import *
from newlang.ast.variable import *
from newlang.memory_tree import *
from newlang.flag import *
from newlang.memory_manager import *

class For_Stat(Node):
    def __init__(self, lower, upper, internal_block, var_name, var_expr, start_is_lower, first_bound, second_bound):
        super().__init__()
        self.lower = lower
        self.upper = upper
        self.internal_block = internal_block
        self.var_name = var_name
        self.var_update = Assignment(Variable(var_name), var_expr)
        self.start_is_lower = start_is_lower
        self.first_bound = first_bound
        self.second_bound = second_bound


    def exec(self, ar):
        lower_val = self.lower.exec(ar)
        upper_val = self.upper.exec(ar)

        start_val = None
        if self.start_is_lower:
            start_val = lower_val if self.first_bound == "<=" else lower_val + 1
        else:
            start_val = upper_val if self.second_bound == "<=" else upper_val - 1

        new_mem = MemoryTree(ar.memory)
        ar.memory = new_mem

        ar.memory[self.var_name] = start_val

        first_check = lower_val < ar.memory[self.var_name] if self.first_bound == "<" else lower_val <= ar.memory[self.var_name]
        second_check = ar.memory[self.var_name] < upper_val if self.second_bound == "<" else ar.memory[self.var_name] <= upper_val
        while first_check and second_check:
            ret = self.internal_block.exec(ar, False)

            if ret is not None:
                first_flag = ret.pop(0)
                if first_flag[0] == Flag.CONTINUE:
                    pass
                elif first_flag[0] == Flag.BREAK:
                    ar.memory = ar.memory.parent_memory

                    if len(ret) > 0:
                        return ret
                    else:
                        return None
                else:
                    ar.memory = ar.memory.parent_memory
                    return ret


            # Update iteration variable
            self.var_update.exec(ar)

            first_check = lower_val < ar.memory[self.var_name] if self.first_bound == "<" else lower_val <= ar.memory[self.var_name]
            second_check = ar.memory[self.var_name] < upper_val if self.second_bound == "<" else ar.memory[self.var_name] <= upper_val


        ar.memory = ar.memory.parent_memory
        return None

    def compile(self, ar, end_function_lbl=None, loop_continue_lbls=None, loop_break_lbls=None):
        if loop_break_lbls is None:
            loop_break_lbls = []
        if loop_continue_lbls is None:
            loop_continue_lbls = []


        ar.mem = MemoryManager.create_new_mem(ar.mem)

        for_label = ar.mem.new_label()
        endfor_label = ar.mem.new_label()
        continue_label = ar.mem.new_label()

        loop_continue_lbls.append(continue_label)
        loop_break_lbls.append(endfor_label)

        command1 = "slt" if self.first_bound == "<" else "sle"
        command2 = "slt" if self.second_bound == "<" else "sle"

        lo_varname = self.lower.compile(ar)
        up_varname = self.upper.compile(ar)

        ar.mem.make_non_evictible(lo_varname)
        ar.mem.make_non_evictible(up_varname)

        lo = ar.mem.reg_name(ar.mem.get_var_in_any_reg(lo_varname))
        up = ar.mem.reg_name(ar.mem.get_var_in_any_reg(up_varname))

        iter_var = ar.mem.reg_name(ar.mem.init_new_var(self.var_name, evictible=False))
        if self.start_is_lower:
            if self.first_bound == "<":
                printc(f"addi {iter_var}, {lo}, 1")
            else:
                printc(f"move {iter_var}, {lo}")
        else:
            if self.second_bound == "<":
                printc(f"addi {iter_var}, {up}, -1")
            else:
                printc(f"move {iter_var}, {up}")

        printc(f"{for_label}:")

        res_varname = ar.mem.get_rand_varname()
        res = ar.mem.reg_name(ar.mem.init_new_var(res_varname, temp=True))
        printc(f"{command1} {res}, {lo}, {iter_var}")
        printc(f"beqz {res}, {endfor_label}")
        printc(f"{command2} {res}, {iter_var}, {up}")
        printc(f"beqz {res}, {endfor_label}")

        ar.mem.delete_var(res_varname)

        self.internal_block.compile(ar, create_new_mem=False, end_function_lbl=end_function_lbl, loop_continue_lbls=loop_continue_lbls, loop_break_lbls=loop_break_lbls)

        printc(f"{continue_label}:")

        # modify var - var modification now is integrated into self.internal_block
        self.var_update.compile(ar)


        printc(f"j {for_label}")

        printc(f"{endfor_label}:")

        ar.mem = MemoryManager.go_back_a_memory(ar.mem)
        # mem.delete_var(lo_varname)
        # mem.delete_var(up_varname)
        # mem.delete_var(self.var_name)
        # mem.delete_var(res_varname)