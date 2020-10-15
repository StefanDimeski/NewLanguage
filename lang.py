import re
import sys
from enum import Enum

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

class If_Statement(Node):
    def __init__(self, cond, block):
        super().__init__()
        self.condition = cond
        self.block = block

    def exec(self, symbol_table):
        condition = self.condition.exec(symbol_table)
        if not isinstance(condition, bool):
            return None

        if condition:
            self.block.exec(symbol_table)

        return None

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
        else:
            print(f'ERROR: Invalid operator type: "{self.op}"')
            return False

class Var(Node):
    def __init__(self, ident):
        super().__init__()
        self.identifier = ident

    def exec(self, symbol_table):
        return symbol_table[self.identifier]

class Number(Node):
    def __init__(self, num):
        super().__init__()
        self.num = int(num)

    def exec(self, symbol_table):
        return self.num

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

regexes = ["^[a-zA-Z][0-9a-zA-Z]*$", "^[0-9][0-9]*$", "^if$", "^print$", "^=$", "^(\+|-)$", "^(\*|/)$", "^(==|!=)$",
           "^\($", "^\)$", "^\{$", "^\}$", "^;$"]

class TType(Enum):
    IDENT = 0
    NUMBER = 1
    IF = 2
    PRINT = 3
    EQUAL = 4
    BIN_OP_PLUS_MIN = 5
    BIN_OP_MULT_DIV = 6
    BIN_OP_EQ_NOTEQ = 7
    OPEN_BRACE = 8
    CLOSED_BRACE = 9
    OPEN_CURLY = 10
    CLOSED_CURLY = 11
    SEMICOLON = 12

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
        return self.parse_block()

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
        else:
            return self.parse_assignment()

    def parse_if_statement(self):
        self.lex.eat(TType.IF)
        self.lex.eat(TType.OPEN_BRACE)

        cond = self.parse_expr()

        self.lex.eat(TType.CLOSED_BRACE)

        block = self.parse_block()

        return If_Statement(cond, block)

    def parse_print_statement(self):
        self.lex.eat(TType.PRINT)
        self.lex.eat(TType.OPEN_BRACE)

        expr = self.parse_expr()

        self.lex.eat(TType.CLOSED_BRACE)

        return Print(expr)

    def parse_assignment(self):
        ident = self.lex.eat(TType.IDENT).val
        self.lex.eat(TType.EQUAL)

        expr = self.parse_expr()

        return Assignment(Var(ident), expr)

    def parse_expr(self):
        node = self.parse_expr1()

        while self.lex.peek().type == TType.BIN_OP_EQ_NOTEQ:
            op = self.lex.eat(TType.BIN_OP_EQ_NOTEQ).val

            node = Operator(node, op, self.parse_expr1())

        return node

    def parse_expr1(self):
        node = self.parse_expr2()

        while self.lex.peek().type == TType.BIN_OP_PLUS_MIN:
            op = self.lex.eat(TType.BIN_OP_PLUS_MIN).val
            node = Operator(node, op, self.parse_expr2())

        return node

    def parse_expr2(self):
        node = self.parse_val()

        while self.lex.peek().type == TType.BIN_OP_MULT_DIV:
            op = self.lex.eat(TType.BIN_OP_MULT_DIV).val
            node = Operator(node, op, self.parse_val())

        return node

    def parse_val(self):
        nxt = self.lex.peek()

        node = None
        if nxt.type == TType.IDENT:
            ident_tok = self.lex.eat(TType.IDENT)
            node = Var(ident_tok.val)
        elif nxt.type == TType.NUMBER:
            num = self.lex.eat(TType.NUMBER).val
            node = Number(num)

        return node


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

    ast.exec(symbol_table)