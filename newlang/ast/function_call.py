from newlang.ast.node import *
from newlang.memory_tree import *
from newlang.activation_record  import *
from newlang.flag import *
from newlang.utils import *
from newlang.ar import *

class Function_Call(Node):
    def __init__(self, name, args, func_node):
        super().__init__()
        self.name = name
        self.args = args
        self.func_node = func_node

    def exec(self, ar):
        if len(self.args) != len(self.func_node.params):
            print(f"FATAL ERROR: '{self.name}' requires {len(self.func_node.params)} parameters, but was given {len(self.args)}")
            exit()

        new_mem = MemoryTree()

        for arg_given, arg_taken in zip(self.args, self.func_node.params):
            new_mem.mem[arg_taken.identifier] = arg_given.exec(ar)

        new_ar = ActivationRecord(new_mem, ar)
        ret = self.func_node.body.exec(new_ar, False)

        del new_mem
        del new_ar

        if ret is None:
            return None

        if len(ret) > 1:
            print(f"ERROR: Invalid sequence of flags inside function call {[a[0] for a in ret]}")
            exit()

        first_item = ret.pop(0)

        if first_item[0] == Flag.RETURN:
            return first_item[1]
        elif first_item[0] == Flag.CONTINUE:
            print(f"ERROR: 'continue' outside of loop")
            exit()
        else:
            print(f"ERROR: 'break' outside of loop")
            exit()

    def compile(self, ar):
        if len(self.args) != len(self.func_node.params):
            print(f"FATAL ERROR: '{self.name}' requires {len(self.func_node.params)} parameters, but was given {len(self.args)}")
            exit()

        ar = AR(ar)
        ar.mem = ar.parent.mem

        #ar.save_vars_on_stack()

        arg_varnames = []

        for idx, arg in enumerate(self.args):
            arg_varname = arg.compile(ar)
            arg_varnames.append(arg_varname)

        # Should it be here? Most likely yes!
        ar.save_vars_on_stack()

        printc(f"addi $sp, $sp, {-4*len(self.args)}")
        for idx, arg in enumerate(self.args):
            printc(f"sw {ar.mem.reg_name(ar.mem.get_var_in_any_reg(arg_varnames[idx]))}, {4*idx}($sp)")


        printc(f"jal {self.func_node.func_lbl}")

        # Invalidate all variables because we don't know what will happen to the registers inside the function
        #ar.invalidate_all_vars()
        ar.reload_saved_vals()

        ar = ar.parent

        for arg_varname in arg_varnames:
            if ar.mem.is_var_temp(arg_varname):
                ar.mem.delete_var(arg_varname)

        # Remove arguments from the stack
        printc(f"addi $sp, $sp, {4*len(self.args)}")

        reg_varname = ar.mem.get_rand_varname()
        reg_idx = ar.mem.init_new_var(reg_varname, temp=True)
        printc(f"move {ar.mem.reg_name(reg_idx)}, $v0")

        return reg_varname