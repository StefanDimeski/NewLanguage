from newlang.utils import *
from newlang.reg_stat import *
from newlang.var import *

import random

class MemoryManager():
    varname_gen = varname_generator()
    label_gen = label_generator()
    register_names = [f"$t{i}" for i in range(0, 10)] + [f"$s{i}" for i in range(0, 8)]

    def __init__(self, parent=None):
        self.parent = parent
        self.free_stack_offset = -4

        self.vars = {}
        self.registers_status = [REG_STAT.FREE for reg in MemoryManager.register_names]
        self.register_varname = [None for reg in MemoryManager.register_names]

    @staticmethod
    def create_new_mem(parent):
        new = MemoryManager(parent)

        new.vars = parent.vars.copy()
        new.registers_status = parent.registers_status
        new.register_varname = parent.register_varname

        return new

    @staticmethod
    def go_back_a_memory(mem_to_go_back_from):
        parent = mem_to_go_back_from.parent

        parent.registers_status = [REG_STAT.FREE for reg in MemoryManager.register_names]
        parent.register_varname = [None for reg in MemoryManager.register_names]

        for varobj in parent.vars.values():
            if varobj.reg is not None:
                parent.registers_status[varobj.reg] = REG_STAT.OCCUPIED
                parent.register_varname[varobj.reg] = varobj.ident

        return parent


    def init_new_var(self, ident, temp=False, evictible=True):
        reg_idx = self.get_free_register()
        self.vars[ident] = Var(ident, reg_idx)
        self.register_varname[reg_idx] = ident
        self.registers_status[reg_idx] = REG_STAT.OCCUPIED
        self.vars[ident].temp = temp
        self.vars[ident].evictible = evictible

        return reg_idx

    def init_new_var_on_stack(self, ident, temp=False):
        offset = self.move_stack_up()
        self.vars[ident] = Var(ident)
        self.vars[ident].temp = temp
        self.vars[ident].stack = offset

        return offset

    def delete_var(self, ident):
        if self.vars[ident].reg is not None:
            self.registers_status[self.vars[ident].reg] = REG_STAT.FREE
            self.register_varname[self.vars[ident].reg] = None

        del self.vars[ident]

    def change_varname(self, old_varname, new_varname, make_non_temp=False):
        reg = self.vars[old_varname].reg
        stack = self.vars[old_varname].stack
        del self.vars[old_varname]

        self.register_varname[reg] = new_varname
        self.vars[new_varname] = Var(new_varname, reg)
        self.vars[new_varname].stack = stack

        if make_non_temp:
            self.vars[new_varname].temp = False

    def make_non_evictible(self, varname):
        self.vars[varname].evictible = False

    def get_var_in_any_reg(self, varname):
        if varname not in self.vars:
            print(f"ERROR: Variable {varname} currently not accounted for!")
            exit()

        if self.vars[varname].reg is not None:
            return self.vars[varname].reg

        free_reg_idx = self.get_free_register()
        self.registers_status[free_reg_idx] = REG_STAT.OCCUPIED
        self.vars[varname].reg = free_reg_idx
        self.register_varname[free_reg_idx] = varname

        printc(f"lw {self.reg_name(free_reg_idx)}, {self.vars[varname].stack}($fp)")

        return free_reg_idx


    def get_free_register(self):
        for idx, reg_stat in enumerate(self.registers_status):
            if reg_stat == REG_STAT.FREE:
                return idx

        # No empty registers, we have to evict one
        rand_idx = random.randrange(0, len(MemoryManager.register_names))
        while not self.vars[self.register_varname[rand_idx]].evictible:
            rand_idx = random.randrange(0, len(MemoryManager.register_names))

        curr_var = self.register_varname[rand_idx]

        self.store_var_on_stack(curr_var)
        self.vars[curr_var].reg = None
        self.register_varname[rand_idx] = None

        return rand_idx

    def is_var_temp(self, ident):
        return self.vars[ident].temp

    def store_var_on_stack(self, varname):
        if self.vars[varname].stack is None:
            offset = self.move_stack_up()
            reg_idx = self.vars[varname].reg
            self.vars[varname].stack = offset
            printc(f"sw {MemoryManager.register_names[reg_idx]}, {self.vars[varname].stack}($fp)")

            self.vars[varname].dirty = False
        elif self.vars[varname].dirty:
            reg_idx = self.vars[varname].reg
            printc(f"sw {MemoryManager.register_names[reg_idx]}, {self.vars[varname].stack}($fp)")

            self.vars[varname].dirty = False

    def move_stack_up(self):
        printc(f"addi $sp, $sp, -4")
        self.free_stack_offset -= 4
        return self.free_stack_offset

    def reg_name(self, reg_idx):
        return MemoryManager.register_names[reg_idx]

    def get_rand_varname(self):
        return next(MemoryManager.varname_gen)

    def new_label(self):
        return next(MemoryManager.label_gen)