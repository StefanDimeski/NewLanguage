import re
import sys
from enum import Enum

class Node:
    def __init__(self):
        self.children = []
        self.terminal = False
        self.reduceable = False

    def initialize(self, value):
        pass

    def add_children(self, children):
        self.children.extend(children)

    def add_child(self, child):
        self.children.append(child)

    def exec_self(self, symbol_table):
        return None

    def exec(self, symbol_table):
        for child in self.children:
            child.exec(symbol_table)

        return self.exec_self(symbol_table)

class Main(Node):
    def __init__(self):
        super().__init__()

class Statement(Node):
    def __init__(self):
        super().__init__()
        self.reduceable = True

class Expression(Node):
    def __init__(self):
        super().__init__()
        self.reduceable = True

class Assignment(Node):
    def __init__(self):
        super().__init__()

    def exec(self, symbol_table):
        ident = self.children[0].identifier
        val = self.children[1].exec(symbol_table)

        symbol_table[ident] = val

        return None

class If_Statement(Node):
    def __init__(self):
        super().__init__()

    def exec(self, symbol_table):
        condition = self.children[0].exec(symbol_table)
        if not isinstance(condition, bool):
            return None

        if condition:
            self.children[1].exec(symbol_table)

        return None

class OperatorExpression(Node):
    def __init__(self):
        super().__init__()

    def exec(self, symbol_table):
        results = []
        for node in self.children:
            results.append(node.exec(symbol_table))

        type = results[1]

        if type == '+':
            return results[0] + results[2]
        elif type == '-':
            return results[0] - results[2]
        elif type == '==':
            return results[0] == results[2]
        elif type == '!=':
            return results[0] != results[2]
        elif type == '*':
            return results[0] * results[2]
        else:
            print(f'ERROR: Invalid operator type: "{type}"')
            return False

class Operator(Node):
    def __init__(self):
        super().__init__()
        self.type = None
        self.terminal = True

    def initialize(self, value):
        self.type = value

    def exec(self, symbol_table):
        return self.type

class Value(Node):
    def __init__(self):
        super().__init__()
        self.value = None
        self.terminal = True

    def initialize(self, value):
        self.value = value

    def exec_self(self, symbol_table):
        return self.value

class Var(Node):
    def __init__(self):
        super().__init__()
        self.identifier = None
        self.terminal = True

    def initialize(self, value):
        self.identifier = value

    def exec_self(self, symbol_table):
        return symbol_table[self.identifier]

class Number(Node):
    def __init__(self):
        super().__init__()
        self.num = None
        self.terminal = True

    def initialize(self, value):
        self.num = int(value)

    def exec_self(self, symbol_table):
        return self.num

class Print(Node):
    def __init__(self):
        super().__init__()

    def exec(self, symbol_table):
        res = self.children[0].exec(symbol_table)

        print(res)
        #if isinstance(res, str):
         #   print(res)
        #else:
         #   print("ERROR: ARGUMENT TO PRINT MUST BE A STRING!")

        return None

class Block(Node):
    def __init__(self):
        super().__init__()

    def exec(self, symbol_table):
        # We create a new table such that we implement scoping
        # By creating a new table, the statements inside the block
        # dont modify the original, thus when out of the block,
        # the variables defined inside the block are discarded
        new_table = symbol_table.copy()
        for child in self.children:
            child.exec(new_table)

        return None

regexes = ["^[a-zA-Z][0-9a-zA-Z]*$", "^[0-9][0-9]*$", "^if$", "^print$", "^=$", "^(\+|-|\*|/|==|!=)$",
           "^\($", "^\)$", "^\{$", "^\}$", "^;$"]

class Token(Enum):
    IDENT = 0
    NUMBER = 1
    IF = 2
    PRINT = 3
    EQUAL = 4
    BIN_OP = 5
    OPEN_BRACE = 6
    CLOSED_BRACE = 7
    OPEN_CURLY = 8
    CLOSED_CURLY = 9
    SEMICOLON = 10

# check out if300 - this is an identifier in C, so it's okay
# but if+ is def not an identifier and it actually is caught only after
# compiling so lexically it's okay, which is what i expected and how this
# lexical analysis built here works

class Lexer():
    def __init__(self, text):
        self.text = text
        self.generator = self.gen_tokens()

    def eat(self, token_type):
        nxt_type, val = next(self.generator)

        if nxt_type != token_type:
            print(f"ERROR: Invalid token {val}")
            exit()

        return (nxt_type, val)

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
                if last_success_token in [0, 1, 5]:
                    yield (Token(last_success_token), inp[:last_success_idx + 1])
                else:
                    yield (Token(last_success_token), None)

                inp = inp[last_success_idx + 1:]
            else:
                print(f'ERROR: Invalid token "{inp}"')
                exit()

def replace(list_name, to_replace, replace_with):
    idx = list_name.index(to_replace)

    list_name = list_name[:idx] + replace_with + list_name[idx + 1:]
    return list_name

def reduce_parse_tree_to_ast(parse_tree_root, parent=None):
    """reduce this node in the tree if it is reducable"""
    ast = parse_tree_root

    children = ast.children.copy()
    for child in children:
        reduce_parse_tree_to_ast(child, ast)

    if ast.reduceable:
        parent.children = replace(parent.children, ast, ast.children)
        #for child in ast.children:
         #   parent.add_child(child)

    return ast

def execute(tree_root):
    symbol_table = {}

    tree_root.exec(symbol_table)

if __name__ == "__main__":

    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as file:
            inp = file.read().replace('\n', '')
            print (inp)
    else:
        inp = input()

    lexer = Lexer(inp)
    print(lexer.eat(Token.IF))