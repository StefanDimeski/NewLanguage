from newlang.ast.node import *
from newlang.utils import *

class Assignment(Node):
    def __init__(self, var, expr):
        super().__init__()
        self.var = var
        self.expr = expr

    def exec(self, ar):
        ident = self.var.identifier
        val = self.expr.exec(ar)

        ar.memory[ident] = val

        return None

    def compile(self, ar):
        mem = ar.mem

        ident = self.var.identifier
        res_name = self.expr.compile(ar)

        if ident not in mem.vars:
            # If we're initializing the new var now, we might as well just use change_varname
            mem.change_varname(res_name, ident, True)
        else:
            # But if we have it already, we must put the new value where the old one was
            if mem.vars[ident].reg is not None:
                printc(f"move {mem.reg_name(mem.vars[ident].reg)}, {mem.reg_name(mem.get_var_in_any_reg(res_name))}")

                if mem.vars[ident].stack is not None:
                    # This indicates that the value in the associated register might be different to the value in the associated stack offset
                    mem.vars[ident].dirty = True

            elif mem.vars[ident].stack is not None:
                printc(f"sw {mem.reg_name(mem.get_var_in_any_reg(res_name))}, {mem.vars[ident].stack}($fp)")
            else:
                print(f"ERROR: Variable '{ident}' exists, but isn't present either in register or in stack!")

        return None