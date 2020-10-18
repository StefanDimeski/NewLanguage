import re
import sys
from enum import Enum
import random
import pyperclip

class AutoNumber(Enum):
    def __new__(cls):
        value = len(cls.__members__)  # note no + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

full_str = ""
def printc(str):
    global full_str
    full_str += (f"{str}\n")
    print(str)



regexes = ["^[a-zA-Z][0-9a-zA-Z]*$", "^[0-9][0-9]*$", "^if$", "^print$", "^else$", "^for$", "^function$", "^=$", "^(\+|-)$", "^(\*|/|%)$", "^(==|!=)$", "^(<|>|<=|>=)$",
           "^\($", "^\)$", "^\{$", "^\}$", "^;$", "^~$"]

class TType(AutoNumber):
    IDENT = ()
    NUMBER = ()
    IF = ()
    PRINT = ()
    ELSE = ()
    FOR = ()
    FUNCTION = ()
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
        reg = self.vars[old_varname].reg
        stack = self.vars[old_varname].stack
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

        printc(f"lw {self.reg_name(free_reg_idx)}, {self.vars[varname].stack}($fp)")

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

        printc(f"sw {self.register_names[self.vars[varname].reg]}, {self.vars[varname].stack}($fp)")

    def move_stack_up(self):
        printc(f"subi $sp, $sp, 4")
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


class ActivationRecord():
    def __init__(self, memorytree, parent):
        self.memory = memorytree
        self.parent = parent

class MemoryTree():
    def __init__(self, parent=None):
        self.parent_memory = parent
        self.mem = {}

    def __getitem__(self, key):
        if key not in self.mem:
            if self.parent_memory is not None:
                return self.parent_memory[key]
            else:
                print(f"ERROR: Uninitialized variable '{key}'")
                exit()

        return self.mem[key]

    def __setitem__(self, key, value):
        if key in self.mem:
            self.mem[key] = value
        else:
            if self.parent_memory is not None:
                if not self.parent_memory.set_var(key, value):
                    self.mem[key] = value

    def set_var(self, key, val):
        """THIS FUNCTION ONLY TO BE USED INTERNALLY"""
        if key in self.mem:
            self.mem[key] = val
            return True

        if self.parent_memory is not None:
            return self.parent_memory.set_var(key, val)

        return False

    def __delitem__(self, key):
        del self.mem[key]

class Node:
    def __init__(self):
        pass

    def exec(self, ar):
        pass

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

    def compile(self, mem):
        ident = self.var.identifier
        res_name = self.expr.compile(mem)

        mem.change_varname(res_name, ident, True)
        return None

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
            self.block.exec(ar)
        else:
            if self.else_block != None:
                self.else_block.exec(ar)

        return None

    def compile(self, mem):
        result_varname = self.condition.compile(mem)
        result_reg = mem.reg_name(mem.get_var_in_any_reg(result_varname))

        else_label = mem.new_label()
        printc(f"beqz {result_reg}, {else_label}")

        if mem.is_var_temp(result_varname):
            mem.delete_var(result_varname)

        self.block.compile(mem)

        if self.else_block == None:
            printc(f"{else_label}:")
        else:
            after_if_label = mem.new_label()
            printc(f"j {after_if_label}")
            printc(f"{else_label}:")
            self.else_block.compile(mem)
            printc(f"{after_if_label}:")

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

    def compile(self, mem):
        res_varname = mem.get_rand_varname()
        free_reg = mem.reg_name(mem.init_new_var(res_varname))

        left_varname = self.left.compile(mem)
        right_varname = self.right.compile(mem)
        type = self.op

        left = mem.reg_name(mem.get_var_in_any_reg(left_varname))
        right = mem.reg_name(mem.get_var_in_any_reg(right_varname))

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

        if not mem.is_var_temp(left_varname):
            mem.delete_var(left_varname)

        if not mem.is_var_temp(right_varname):
            mem.delete_var(right_varname)

        return res_varname

class Variable(Node):
    def __init__(self, ident):
        super().__init__()
        self.identifier = ident

    def exec(self, ar):
        return ar.memory[self.identifier]

    def compile(self, mem):
        #mem.get_var_in_any_reg(self.identifier) #TODO should it just return self.identifier?
        return self.identifier


class Number(Node):
    def __init__(self, num):
        super().__init__()
        self.num = int(num)

    def exec(self, ar):
        return self.num

    def compile(self, mem):
        varname = mem.get_rand_varname()
        reg = mem.reg_name(mem.init_new_var(varname))
        printc(f"li {reg}, {self.num}")
        return varname

class Print(Node):
    def __init__(self, expr):
        super().__init__()
        self.expr = expr

    def exec(self, ar):
        res = self.expr.exec(ar)

        print(res)
        #if isinstance(res, str):
         #   print(res)
        #else:
         #   print("ERROR: ARGUMENT TO PRINT MUST BE A STRING!")

        return None

    def compile(self, mem):
        res_varname = self.expr.compile(mem)
        reg = mem.reg_name(mem.get_var_in_any_reg(res_varname))

        printc(f"li $v0, 1")
        printc(f"move $a0, {reg}")
        printc("syscall")

class Program(Node):
    def __init__(self, funcs, block):
        super().__init__()
        self.funcs = funcs
        self.block = block

    def exec(self, ar):
        self.block.exec(ar)

    def compile(self, mem):
        self.block.compile(mem)

        # end program
        printc("li $v0, 10")
        printc("syscall")

class Function(Node):
    def __init__(self, name, params, body):
        super().__init__()
        self.name = name
        self.params = params
        self.body = body

    def exec(self, ar):
        pass

    def compile(self, mem):
        pass

class Block(Node):
    def __init__(self, children):
        super().__init__()
        self.children = children

    def exec(self, ar, func_call=False):
        if not func_call:
            new_mem = MemoryTree(ar.memory)
            ar.memory = new_mem

        for child in self.children:
            child.exec(ar)

        # Return it back (i.e. discard variables that were created in the scope of the block that just ended)
        if not func_call:
            ar.memory = ar.memory.parent_memory

        return None

    def compile(self, mem):
        for child in self.children:
            child.compile(mem)

class Input(Node):
    def __init__(self):
        super().__init__()
        pass

    def exec(self, ar):
        return int(input())

    def compile(self, mem):
        printc("li $v0, 5")
        printc("syscall")

        varname = mem.get_rand_varname()
        free_reg = mem.reg_name(mem.init_new_var(varname))

        printc(f"move {free_reg}, $v0")

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

    def exec(self, ar):
        lower_val = self.lower.exec(ar)
        upper_val = self.upper.exec(ar)

        start_val = None
        if self.up_down == "+":
            start_val = lower_val if self.first_bound == "<=" else lower_val + 1
        else:
            start_val = upper_val if self.second_bound == "<=" else upper_val - 1

        ar.memory[self.var_name] = start_val

        first_check = lower_val < ar.memory[self.var_name] if self.first_bound == "<" else lower_val <= ar.memory[self.var_name]
        second_check = ar.memory[self.var_name] < upper_val if self.second_bound == "<" else ar.memory[self.var_name] <= upper_val
        while first_check and second_check:
            self.internal_block.exec(ar)

            if self.up_down == "+":
                ar.memory[self.var_name] += 1
            else:
                ar.memory[self.var_name] -= 1

            first_check = lower_val < ar.memory[self.var_name] if self.first_bound == "<" else lower_val <= ar.memory[self.var_name]
            second_check = ar.memory[self.var_name] < upper_val if self.second_bound == "<" else ar.memory[self.var_name] <= upper_val

        del ar.memory[self.var_name]

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
                printc(f"addi {iter_var}, {lo}, 1")
            else:
                printc(f"move {iter_var}, {lo}")
        else:
            if self.second_bound == "<":
                printc(f"subi {iter_var}, {up}, 1")
            else:
                printc(f"move {iter_var}, {up}")

        printc(f"{for_label}:")

        res_varname = mem.get_rand_varname()
        res = mem.reg_name(mem.init_new_var(res_varname))
        printc(f"{command1} {res}, {lo}, {iter_var}")
        printc(f"beqz {res}, {endfor_label}")
        printc(f"{command2} {res}, {iter_var}, {up}")
        printc(f"beqz {res}, {endfor_label}")

        self.internal_block.compile(mem)

        # modify var
        if self.up_down == "+":
            printc(f"addi {iter_var}, {iter_var}, 1")
        else:
            printc(f"subi {iter_var}, {iter_var}, 1")

        printc(f"j {for_label}")

        printc(f"{endfor_label}:")

        mem.delete_var(lo_varname)
        mem.delete_var(up_varname)
        mem.delete_var(self.var_name)
        mem.delete_var(res_varname)

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
        self.func_node.body.exec(new_ar, True)

        del new_mem
        del new_ar

    def compile(self, mem):
        pass

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
        self.next_token = []

    def eat(self, token_type):
        nxt = None
        if len(self.next_token) > 0:
            nxt = self.next_token.pop(0)
        else:
            nxt = next(self.generator)

        if nxt.type != token_type:
            print(f"ERROR: Invalid token {nxt.val}")
            exit()

        return nxt

    def eat_list(self, list_of_types):
        nxt = None
        if len(self.next_token) > 0:
            nxt = self.next_token.pop(0)
        else:
            nxt = next(self.generator)

        if nxt.type not in list_of_types:
            print(f"ERROR: Invalid token {nxt.val}")
            exit()

        return nxt

    def peek(self, num=1):
        if len(self.next_token) >= num:
            return self.next_token[num - 1]

        num_peeks = num - len(self.next_token)
        for i in range(0, num_peeks):
            self.next_token.append(next(self.generator))

        return self.next_token[-1]

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
                        # printc(f"{index}, {regex}")
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
        self.func_name_to_ast_node = {}

    def create_ast(self):
        return self.parse_program()

    def parse_program(self):
        funcs = []
        while self.lex.peek().type == TType.FUNCTION:
            funcs.append(self.parse_function_declaration())

        return Program(funcs, self.parse_block())

    def parse_function_declaration(self):
        self.lex.eat(TType.FUNCTION)

        ident = self.lex.eat(TType.IDENT).val
        self.lex.eat(TType.OPEN_BRACE)

        params = []
        if self.lex.peek().type != TType.CLOSED_BRACE:
            params.append(Variable(self.lex.eat(TType.IDENT).val))

        while self.lex.peek().type != TType.CLOSED_BRACE:
            self.lex.eat(TType.SEMICOLON)
            params.append(Variable(self.lex.eat(TType.IDENT).val))

        self.lex.eat(TType.CLOSED_BRACE)

        body = self.parse_block()

        node = Function(ident, params, body)
        self.func_name_to_ast_node[ident] = node

        return node

    def parse_function_call(self):
        name = self.lex.eat(TType.IDENT).val
        self.lex.eat(TType.OPEN_BRACE)

        args = []
        if self.lex.peek().type != TType.CLOSED_BRACE:
            args.append(self.parse_expr1())

        while self.lex.peek().type != TType.CLOSED_BRACE:
            self.lex.eat(TType.SEMICOLON)
            args.append(self.parse_expr1())

        self.lex.eat(TType.CLOSED_BRACE)

        return Function_Call(name, args, self.func_name_to_ast_node[name])


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
        elif nxt.type == TType.IDENT:
            if self.lex.peek(2).type == TType.OPEN_BRACE:
                return self.parse_function_call()
            else:
                return self.parse_assignment()
        else:
            print(f"ERROR: Unexpected token {nxt.val}")
            exit()

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
    mem = MemoryTree()
    ar_main = ActivationRecord(mem, None)

    tree_root.exec(ar_main)

def compile(ast):
    mem = MemoryManager()
    ast.compile(mem)
    pyperclip.copy(full_str)

if __name__ == "__main__":

    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as file:
            inp = file.read().replace('\n', '')
    else:
        inp = input()

    lexer = Lexer(inp)
    parser = Parser(lexer)

    ast = parser.create_ast()


    execute(ast)
    #compile(ast)