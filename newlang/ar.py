from newlang.reg_stat import *
from newlang.memory_manager import *
from newlang.utils import *

class AR():
    def __init__(self, parent=None):
        self.mem = None
        self.parent = parent

        self.saved_vars = []

    def save_vars_on_stack(self):
        for var in self.mem.vars.values():
            # In saved vars we put all the vars that we need to restore to registers after the call
            # therefore only if the variable is in a register do we append it to the list for restoration
            if var.reg is not None:
                self.saved_vars.append(var)

                if var.stack is None or var.dirty:
                    self.mem.store_var_on_stack(var.ident)


    def invalidate_all_vars(self):
        # Invalidate it so that next time it's used it has to be picked up from the stack
        for var in self.mem.vars.values():
            if var.reg is not None:
                self.mem.register_varname[var.reg] = None
                self.mem.registers_status[var.reg] = REG_STAT.FREE
                var.reg = None

    def reload_saved_vals(self):
        for i in range(len(self.mem.registers_status)):
            self.mem.register_varname[i] = None
            self.mem.registers_status[i] = REG_STAT.FREE

        for var in self.saved_vars:
            printc(f"lw {self.mem.reg_name(var.reg)}, {var.stack}($fp)")
            self.mem.registers_status[var.reg] = REG_STAT.OCCUPIED
            self.mem.register_varname[var.reg] = var.ident

        self.saved_vars.clear()

    def init(self):
        printc(f"addi $sp, $sp, -8")
        printc(f"sw $ra, 4($sp)")
        printc(f"sw $fp, 0($sp)")
        printc(f"addi $fp, $sp, 4")

        self.mem = MemoryManager()

    def add_argument(self, name, arg_reg):
        offset_to_store = self.mem.init_var_on_stack(name)
        printc(f"sw ${self.mem.reg_name(arg_reg)}, {offset_to_store}($fp)")

    def destroy(self):
        printc(f"addi $sp, $fp, 4")
        printc(f"lw $ra, 0($fp)")
        printc(f"lw $fp, -4($fp)")