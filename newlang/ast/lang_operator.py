from newlang.ast.node import *
from newlang.utils import *

class Operator(Node):
    def __init__(self, left, op, right):
        super().__init__()
        self.left = left
        self.right = right
        self.op = op

    def exec(self, ar):
        if self.op == '+':
            return int(self.left.exec(ar)) + int(self.right.exec(ar))
        elif self.op == '-':
            return int(self.left.exec(ar)) - int(self.right.exec(ar))
        elif self.op == '==':
            return int(self.left.exec(ar)) == int(self.right.exec(ar))
        elif self.op == '!=':
            return int(self.left.exec(ar)) != int(self.right.exec(ar))
        elif self.op == '*':
            return int(self.left.exec(ar)) * int(self.right.exec(ar))
        elif self.op == '/':
            return int(self.left.exec(ar)) / int(self.right.exec(ar))
        elif self.op == '%':
            return int(self.left.exec(ar)) % int(self.right.exec(ar))
        elif self.op == '<':
            return int(self.left.exec(ar)) < int(self.right.exec(ar))
        elif self.op == '<=':
            return int(self.left.exec(ar)) <= int(self.right.exec(ar))
        elif self.op == '>':
            return int(self.left.exec(ar)) > int(self.right.exec(ar))
        elif self.op == '>=':
            return int(self.left.exec(ar)) >= int(self.right.exec(ar))
        else:
            print(f'ERROR: Invalid operator type: "{self.op}"')
            return False

    def compile(self, ar):
        mem = ar.mem

        left_varname = self.left.compile(ar)
        right_varname = self.right.compile(ar)
        type = self.op

        left = mem.reg_name(mem.get_var_in_any_reg(left_varname))
        right = mem.reg_name(mem.get_var_in_any_reg(right_varname))

        res_varname = mem.get_rand_varname()
        free_reg = mem.reg_name(mem.init_new_var(res_varname, temp=True))

        if type == "+":
            printc(f"add {free_reg}, {left}, {right}")
        elif type == '-':
            printc(f"sub {free_reg}, {left}, {right}")
        elif type == '*':
            printc(f"mul {free_reg}, {left}, {right}")
        elif type == '/':
            printc(f"div {free_reg}, {left}, {right}")
        elif type == '%':
            printc(f"div {left}, {right}")
            printc(f"mfhi {free_reg}")
        elif type == '==':
            printc(f"seq {free_reg}, {left}, {right}")
        elif type == '!=':
            printc(f"sne {free_reg}, {left}, {right}")
        elif self.op == '<':
            printc(f"slt {free_reg}, {left}, {right}")
        elif self.op == '<=':
            printc(f"sle {free_reg}, {left}, {right}")
        elif self.op == '>':
            printc(f"sgt {free_reg}, {left}, {right}")
        elif self.op == '>=':
            printc(f"sge {free_reg}, {left}, {right}")
        else:
            print(f"ERROR: Invalid operator {type}")

        if mem.is_var_temp(left_varname):
            mem.delete_var(left_varname)

        if mem.is_var_temp(right_varname):
            mem.delete_var(right_varname)

        return res_varname