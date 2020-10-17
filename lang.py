import re
import sys
from enum import Enum
import random

regexes = ["^[a-zA-Z][0-9a-zA-Z]*$", "^[0-9][0-9]*$", "^if$", "^print$", "^else$", "^for$", "^=$", "^(\+|-)$", "^(\*|/|%)$", "^(==|!=)$", "^(<|>|<=|>=)$",
           "^\($", "^\)$", "^\{$", "^\}$", "^;$", "^~$"]


class AutoNumber(Enum):
    def __new__(cls):
        value = len(cls.__members__)  # note no + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

class TType(AutoNumber):
    IDENT = ()
    NUMBER = ()
    IF = ()
    PRINT = ()
    ELSE = ()
    FOR = ()
    EQUAL = ()
    BIN_OP_PLUS_MIN = ()
    BIN_OP_MULT_DIV = ()
    BIN_OP_EQ_NOTEQ = ()
    BIN_OP_GREATER_SMALLER = ()
    OPEN_BRACE = ()
    CLOSED_BRACE = ()
    OPEN_CURLY = ()
    CLOSED_CURLY = ()
    SEMICOLON = ()
    TILDE = ()

class REG_STAT(Enum):
    OCCUPIED = 0
    DIRTY = 1
    FREE = 2

class Var():
    def __init__(self, ident, reg=None):
        self.reg = reg
        self.stack = None
        self.ident = ident
        self.temp = True

class MemoryManager():
    def __init__(self):
        self.register_names = [f"$t{i}" for i in range(0, 8)]
        self.free_stack_offset = 0
        self.varname_gen = self.varname_generator()
        self.label_gen = self.label_generator()

        self.vars = {}
        self.registers_status = [REG_STAT.FREE for reg in self.register_names]
        self.register_varname = [None for reg in self.register_names]

    def init_new_var(self, ident, temp=True):
        reg_idx = self.get_free_register()
        self.vars[ident] = Var(ident, reg_idx)
        self.register_varname[reg_idx] = ident
        self.registers_status[reg_idx] = REG_STAT.OCCUPIED
        self.vars[ident].temp = temp

        return reg_idx

    def delete_var(self, ident):
        self.registers_status[self.vars[ident].reg] = REG_STAT.FREE
        self.register_varname[self.vars[ident].reg] = None

        del self.vars[ident]

    def change_varname(self, old_varname, new_varname, make_non_temp=False):
        reg = self.vars[old_varname]
        stack = self.vars[old_varname]
        del self.vars[old_varname]

        self.register_varname[reg] = new_varname
        self.vars[new_varname] = Var(new_varname, reg)
        self.vars[new_varname].stack = stack

        if make_non_temp:
            self.vars[new_varname].temp = False

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

        print(f"lw {self.reg_name(free_reg_idx)}, {self.vars[varname].stack}($fp)")

        return free_reg_idx


    def get_free_register(self):
        for idx, reg_stat in enumerate(self.registers_status):
            if reg_stat == REG_STAT.FREE:
                return idx

        # No empty registers, we have to convict one
        rand_idx = random.randrange(0, len(self.register_names))
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

        print(f"sw {self.register_names[self.vars[varname].reg]}, {self.vars[varname].stack}($fp)")

    def move_stack_up(self):
        print(f"subi $sp, $sp, 4")
        self.free_stack_offset -= 4
        return self.free_stack_offset

    def reg_name(self, reg_idx):
        return self.register_names[reg_idx]

    def get_rand_varname(self):
        return next(self.varname_gen)

    def new_label(self):
        return next(self.label_gen)

    def label_generator(self):
        num = 0
        while True:
            num_str = str(num)
            label = ""

            for digit in num_str:
                label += chr(97 + int(digit))

            num += 1
            yield label

    def varname_generator(self):
        num = 0
        while True:
            num_str = str(num)
            label = "__"

            for digit in num_str:
                label += chr(97 + int(digit))

            num += 1
            yield label

class Node:
    def __init__(self):
        pass

    def exec(self, symbol_table):
        pass

class Assignment(Node):
    def __init__(self, var, expr):
        super().__init__()
        self.var = var
        self.expr = expr

    def exec(self, symbol_table):
        ident = self.var.identifier
        val = self.expr.exec(symbol_table)

        symbol_table[ident] = val

        return None

    def compile(self, mem):
        ident = self.var.identifier
        res_name = self.expr.compile(mem)

        mem.change_varname(res_name, ident)
        return None

class If_Statement(Node):
    def __init__(self, cond, block, else_block):
        super().__init__()
        self.condition = cond
        self.block = block
        self.else_block = else_block

    def exec(self, symbol_table):
        condition = self.condition.exec(symbol_table)
        if not isinstance(condition, bool):
            return None

        if condition:
            self.block.exec(symbol_table)
        else:
            if self.else_block != None:
                self.else_block.exec(symbol_table)

        return None

    def compile(self, mem):
        result_varname = self.condition.compile(mem)
        result_reg = mem.reg_name(mem.get_var_in_any_reg(result_varname))

        else_label = mem.new_label()
        print(f"beqz {result_reg}, {else_label}")

        if mem.is_var_temp(result_varname):
            mem.delete_var(result_varname)

        self.block.compile(mem)

        if self.else_block == None:
            print(f"{else_label}:")
        else:
            after_if_label = mem.new_label()
            print(f"j {after_if_label}")
            print(f"{else_label}:")
            self.else_block.compile(mem)
            print(f"{after_if_label}:")

class Operator(Node):
    def __init__(self, left, op, right):
        super().__init__()
        self.left = left
        self.right = right
        self.op = op

    def exec(self, symbol_table):
        if self.op == '+':
            return int(self.left.exec(symbol_table)) + int(self.right.exec(symbol_table))
        elif self.op == '-':
            return int(self.left.exec(symbol_table)) - int(self.right.exec(symbol_table))
        elif self.op == '==':
            return int(self.left.exec(symbol_table)) == int(self.right.exec(symbol_table))
        elif self.op == '!=':
            return int(self.left.exec(symbol_table)) != int(self.right.exec(symbol_table))
        elif self.op == '*':
            return int(self.left.exec(symbol_table)) * int(self.right.exec(symbol_table))
        elif self.op == '/':
            return int(self.left.exec(symbol_table)) / int(self.right.exec(symbol_table))
        elif self.op == '%':
            return int(self.left.exec(symbol_table)) % int(self.right.exec(symbol_table))
        elif self.op == '<':
            return int(self.left.exec(symbol_table)) < int(self.right.exec(symbol_table))
        elif self.op == '<=':
            return int(self.left.exec(symbol_table)) <= int(self.right.exec(symbol_table))
        elif self.op == '>':
            return int(self.left.exec(symbol_table)) > int(self.right.exec(symbol_table))
        elif self.op == '>=':
            return int(self.left.exec(symbol_table)) >= int(self.right.exec(symbol_table))
        else:
            print(f'ERROR: Invalid operator type: "{self.op}"')
            return False

    def compile(self, mem):
        res_varname = mem.get_rand_varname()
        free_reg = mem.reg_name(mem.init_new_var(res_varname))

        left_varname = self.left.compile(mem)
        right_varname = self.right.compile(mem)
        type = self.op

        left = mem.reg_name(mem.get_var_in_any_reg(left_varname))
        right = mem.reg_name(mem.get_var_in_any_reg(right_varname))

        if type == "+":
            print(f"add {free_reg}, {left}, {right}")
        elif type == '-':
            print(f"sub {free_reg}, {left}, {right}")
        elif type == '*':
            print(f"mul {free_reg}, {left}, {right}")
        elif type == '/':
            print(f"div {free_reg}, {left}, {right}")
        elif type == '%':
            print(f"div {left}, {right}")
            print(f"mfhi {free_reg}")
        elif type == '==':
            print(f"seq {free_reg}, {left}, {right}")
        elif type == '!=':
            print(f"sne {free_reg}, {left}, {right}")
        elif self.op == '<':
            print(f"slt {free_reg}, {left}, {right}")
        elif self.op == '<=':
            print(f"sle {free_reg}, {left}, {right}")
        elif self.op == '>':
            print(f"sgt {free_reg}, {left}, {right}")
        elif self.op == '>=':
            print(f"sge {free_reg}, {left}, {right}")
        else:
            print(f"ERROR: Invalid operator {type}")

        if not mem.is_var_temp(left_varname):
            mem.delete_var(left_varname)

        if not mem.is_var_temp(right_varname):
            mem.delete_var(right_varname)

        return res_varname

class Variable(Node):
    def __init__(self, ident):
        super().__init__()
        self.identifier = ident

    def exec(self, symbol_table):
        return symbol_table[self.identifier]

    def compile(self, mem):
        #mem.get_var_in_any_reg(self.identifier) #TODO should it just return self.identifier?
        return self.identifier


class Number(Node):
    def __init__(self, num):
        super().__init__()
        self.num = int(num)

    def exec(self, symbol_table):
        return self.num

    def compile(self, mem):
        varname = mem.get_rand_varname()
        reg = mem.reg_name(mem.init_new_var(varname))
        print(f"li {reg}, {self.num}")
        return varname

class Print(Node):
    def __init__(self, expr):
        super().__init__()
        self.expr = expr

    def exec(self, symbol_table):
        res = self.expr.exec(symbol_table)

        print(res)
        #if isinstance(res, str):
         #   print(res)
        #else:
         #   print("ERROR: ARGUMENT TO PRINT MUST BE A STRING!")

        return None

    def compile(self, mem):
        res_varname = self.expr.compile(mem)
        reg = mem.reg_name(mem.get_var_in_any_reg(res_varname))

        print(f"li $v0, 1")
        print(f"move $a0, {reg}")
        print("syscall")

class Program(Node):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def exec(self, symbol_table):
        self.block.exec(symbol_table)

    def compile(self, mem):
        self.block.compile(mem)

        # end program
        print("li $v0, 10")
        print("syscall")

class Block(Node):
    def __init__(self, children):
        super().__init__()
        self.children = children

    def exec(self, symbol_table):
        # We create a new table such that we implement scoping
        # By creating a new table, the statements inside the block
        # dont modify the original, thus when out of the block,
        # the variables defined inside the block are discarded
        new_table = symbol_table.copy()
        for child in self.children:
            child.exec(new_table)

        return None

    def compile(self, mem):
        for child in self.children:
            child.compile(mem)

class Input(Node):
    def __init__(self):
        super().__init__()
        pass

    def exec(self, symbol_table):
        return int(input())

    def compile(self, mem):
        print("li $v0, 5")
        print("syscall")

        varname = mem.get_rand_varname()
        free_reg = mem.reg_name(mem.init_new_var(varname))

        print(f"move {free_reg}, $v0")

        return varname

class For_Stat(Node):
    def __init__(self, lower, upper, internal_block, var_name, up_down, first_bound, second_bound):
        super().__init__()
        self.lower = lower
        self.upper = upper
        self.internal_block = internal_block
        self.var_name = var_name
        self.up_down = up_down
        self.first_bound = first_bound
        self.second_bound = second_bound

    def exec(self, symbol_table):
        lower_val = self.lower.exec(symbol_table)
        upper_val = self.upper.exec(symbol_table)

        start_val = None
        if self.up_down == "+":
            start_val = lower_val if self.first_bound == "<=" else lower_val + 1
        else:
            start_val = upper_val if self.second_bound == "<=" else upper_val - 1

        symbol_table[self.var_name] = start_val

        first_check = lower_val < symbol_table[self.var_name] if self.first_bound == "<" else lower_val <= symbol_table[self.var_name]
        second_check = symbol_table[self.var_name] < upper_val if self.second_bound == "<" else symbol_table[self.var_name] <= upper_val
        while first_check and second_check:
            self.internal_block.exec(symbol_table)

            if self.up_down == "+":
                symbol_table[self.var_name] += 1
            else:
                symbol_table[self.var_name] -= 1

            first_check = lower_val < symbol_table[self.var_name] if self.first_bound == "<" else lower_val <= symbol_table[self.var_name]
            second_check = symbol_table[self.var_name] < upper_val if self.second_bound == "<" else symbol_table[self.var_name] <= upper_val

        del symbol_table[self.var_name]

        return None

    def compile(self, mem):
        for_label = mem.new_label()
        endfor_label = mem.new_label()

        command1 = "slt" if self.first_bound == "<" else "sle"
        command2 = "slt" if self.second_bound == "<" else "sle"

        lo_varname = self.lower.compile(mem)
        up_varname = self.upper.compile(mem)

        lo = mem.reg_name(mem.get_var_in_any_reg(lo_varname))
        up = mem.reg_name(mem.get_var_in_any_reg(up_varname))

        # mem.store_reg_in_memory_as(lo, "FOR.lo")
        # mem.store_reg_in_memory_as(up, "FOR.up")
        #
        # mem.free_register(lo)
        # mem.free_register(hi)

        iter_var = mem.reg_name(mem.init_new_var(self.var_name))
        if self.up_down == "+":
            if self.first_bound == "<":
                print(f"addi {iter_var}, {lo}, 1")
            else:
                print(f"move {iter_var}, {lo}")
        else:
            if self.second_bound == "<":
                print(f"subi {iter_var}, {up}, 1")
            else:
                print(f"move {iter_var}, {up}")

        print(f"{for_label}:")

        res_varname = mem.get_rand_varname()
        res = mem.reg_name(mem.init_new_var(res_varname))
        print(f"{command1} {res}, {lo}, {iter_var}")
        print(f"beqz {res}, {endfor_label}")
        print(f"{command2} {res}, {iter_var}, {up}")
        print(f"beqz {res}, {endfor_label}")

        self.internal_block.compile(mem)

        # modify var
        if self.up_down == "+":
            print(f"addi {iter_var}, {iter_var}, 1")
        else:
            print(f"subi {iter_var}, {iter_var}, 1")

        print(f"j {for_label}")

        print(f"{endfor_label}:")

        mem.delete_var(lo_varname)
        mem.delete_var(up_varname)
        mem.delete_var(self.var_name)
        mem.delete_var(res_varname)

class Token():
    def __init__(self, type, val):
        self.type = type
        self.val = val
# check out if300 - this is an identifier in C, so it's okay
# but if+ is def not an identifier and it actually is caught only after
# compiling so lexically it's okay, which is what i expected and how this
# lexical analysis built here works

class Lexer():
    def __init__(self, text):
        self.text = text
        self.generator = self.gen_tokens()
        self.next_token = None

    def eat(self, token_type):
        nxt = None
        if self.next_token != None:
            nxt = self.next_token
            self.next_token = None
        else:
            nxt = next(self.generator)

        if nxt.type != token_type:
            print(f"ERROR: Invalid token {nxt.val}")
            exit()

        return nxt

    def eat_list(self, list_of_types):
        nxt = None
        if self.next_token != None:
            nxt = self.next_token
            self.next_token = None
        else:
            nxt = next(self.generator)

        if nxt.type not in list_of_types:
            print(f"ERROR: Invalid token {nxt.val}")
            exit()

        return nxt

    def peek(self):
        if self.next_token != None:
            return self.next_token

        self.next_token = next(self.generator)
        return self.next_token

    def gen_tokens(self):
        inp = self.text
        while len(inp) > 0:

            while inp[0] == " ":
                inp = inp[1:]

            last_success_idx = -1
            last_success_token = -1
            for a in range(0, len(inp)):
                substr = inp[:a + 1]
                at_least_one = False
                for index, regex in enumerate(regexes):
                    if re.search(regex, substr):
                        # print(f"{index}, {regex}")
                        at_least_one = True
                        last_success_idx = a
                        last_success_token = index

                if not at_least_one and last_success_idx != -1:
                    break

            if last_success_token != -1:
                yield Token(TType(last_success_token), inp[:last_success_idx + 1])

                inp = inp[last_success_idx + 1:]
            else:
                print(f'ERROR: Invalid token "{inp}"')
                exit()

class Parser():
    def __init__(self, lexer):
        self.lex = lexer

    def create_ast(self):
        return self.parse_program()

    def parse_program(self):
        return Program(self.parse_block())

    def parse_block(self):
        self.lex.eat(TType.OPEN_CURLY)

        children = []
        children.append(self.parse_statement())

        while self.lex.peek().type == TType.SEMICOLON:
            self.lex.eat(TType.SEMICOLON)
            children.append(self.parse_statement())

        self.lex.eat(TType.CLOSED_CURLY)

        return Block(children)

    def parse_statement(self):
        nxt = self.lex.peek()

        if nxt.type == TType.OPEN_CURLY:
            return self.parse_block()
        elif nxt.type == TType.IF:
            return self.parse_if_statement()
        elif nxt.type == TType.PRINT:
            return self.parse_print_statement()
        elif nxt.type == TType.TILDE:
            return self.parse_input()
        elif nxt.type == TType.FOR:
            return self.parse_for_statement()
        else:
            return self.parse_assignment()

    def parse_if_statement(self):
        self.lex.eat(TType.IF)
        self.lex.eat(TType.OPEN_BRACE)

        cond = self.parse_expr()

        self.lex.eat(TType.CLOSED_BRACE)

        block = self.parse_block()

        else_block = None
        if self.lex.peek().type == TType.ELSE:
            self.lex.eat(TType.ELSE)
            else_block = self.parse_block()

        return If_Statement(cond, block, else_block)

    def parse_print_statement(self):
        self.lex.eat(TType.PRINT)
        self.lex.eat(TType.OPEN_BRACE)

        expr = self.parse_expr()

        self.lex.eat(TType.CLOSED_BRACE)

        return Print(expr)

    def parse_for_statement(self):
        self.lex.eat(TType.FOR)
        lower = self.parse_expr1()

        a = self.lex.eat(TType.BIN_OP_GREATER_SMALLER)
        if a.val not in ["<", "<="]:
            print(f"ERROR: Invalid syntax. Unexpected token {a.val}")
            exit()

        var_name = self.lex.eat(TType.IDENT)
        up_down = self.lex.eat(TType.BIN_OP_PLUS_MIN)

        b = self.lex.eat(TType.BIN_OP_GREATER_SMALLER)
        if b.val not in ["<", "<="]:
            print(f"ERROR: Invalid syntax. Unexpected token {b.val}")
            exit()

        upper = self.parse_expr1()

        internal_block = self.parse_block()

        return For_Stat(lower, upper, internal_block, var_name.val, up_down.val, a.val, b.val)

    def parse_assignment(self):
        ident = self.lex.eat(TType.IDENT).val
        self.lex.eat(TType.EQUAL)

        expr = self.parse_expr()

        return Assignment(Variable(ident), expr)

    def parse_expr(self):
        node = self.parse_expr1()

        while self.lex.peek().type in [TType.BIN_OP_EQ_NOTEQ, TType.BIN_OP_GREATER_SMALLER]:
            op = self.lex.eat_list([TType.BIN_OP_EQ_NOTEQ, TType.BIN_OP_GREATER_SMALLER]).val
            node = Operator(node, op, self.parse_expr1())

        return node

    def parse_expr1(self):
        node = self.parse_expr2()

        while self.lex.peek().type == TType.BIN_OP_PLUS_MIN:
            op = self.lex.eat(TType.BIN_OP_PLUS_MIN).val

            next_expr = self.parse_expr2()

            if isinstance(node, Number) and isinstance(next_expr, Number):
                node = Number(node.num + next_expr.num) if op == "+" else Number(node.num - next_expr.num)
            else:
                node = Operator(node, op, next_expr)

        return node

    def parse_expr2(self):
        node = self.parse_val()

        while self.lex.peek().type == TType.BIN_OP_MULT_DIV:
            op = self.lex.eat(TType.BIN_OP_MULT_DIV).val

            next_expr = self.parse_val()

            if isinstance(node, Number) and isinstance(next_expr, Number):
                if op == "*":
                    node = Number(node.num * next_expr.num)
                elif op == "/":
                    node = Number(node.num / next_expr.num)
                elif op == "%":
                    node = Number(node.num % next_expr.num)
            else:
                node = Operator(node, op, next_expr)

        return node

    def parse_val(self):
        nxt = self.lex.peek()

        node = None
        if nxt.type == TType.OPEN_BRACE:
            self.lex.eat(TType.OPEN_BRACE)
            node = self.parse_expr()
            self.lex.eat(TType.CLOSED_BRACE)
        if nxt.type == TType.IDENT:
            ident_tok = self.lex.eat(TType.IDENT)
            node = Variable(ident_tok.val)
        elif nxt.type == TType.NUMBER:
            num = self.lex.eat(TType.NUMBER).val
            node = Number(num)
        elif nxt.type == TType.TILDE:
            node = self.parse_input()

        return node

    def parse_input(self):
        self.lex.eat(TType.TILDE)

        return Input()


def execute(tree_root):
    symbol_table = {}

    tree_root.exec(symbol_table)

if __name__ == "__main__":

    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as file:
            inp = file.read().replace('\n', '')
    else:
        inp = input()

    lexer = Lexer(inp)
    parser = Parser(lexer)

    ast = parser.create_ast()

    symbol_table = {}
    mem = MemoryManager()

    #ast.exec(symbol_table)
    ast.compile(mem)