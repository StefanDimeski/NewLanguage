import re
import sys

class MemoryManager():
    def __init__(self):
        # Both of these dictionaries are indexed by symbol
        self.register_table = {}
        self.stack_table = {} # offset from $fp
        self.registers = [f"$t{i}" for i in range(0, 8)]
        self.register_occupied = [False for i in range(0, 8)]
        self.free_offset = 0

        print("move $fp, $sp")

    def load_symbol_in_register(self, symbol_ident, register):
        if symbol_ident in self.register_table:
            return self.register_table[symbol_ident]

        if symbol_ident in self.stack_table:
            print(f"lw {register}, {self.stack_table[symbol_ident]}($fp)")
            return register

        self.occupy_reg(register)

        return None


    def load_symbol_in_free_register(self, symbol_ident):
        free_reg = self.get_free_register()

        return self.load_symbol_in_register(symbol_ident, free_reg)

    def stack_up(self):
        print(f"addi $sp, $sp, -4")

    def store_ident_in_memory(self, id):
        self.stack_up()

        reg = self.register_table[id]

        print(f"sw {reg}, {self.free_offset}($fp)")
        self.stack_table[id] = self.free_offset

        self.free_offset -= 4

    def get_free_register(self, occupy=True):
        for idx, r in enumerate(self.register_occupied):
            if not r:
                if occupy:
                    self.occupy_reg(self.registers[idx])
                return self.registers[idx]

        # At this point no register was free so we need to evict one TODO
        
    def occupy_reg(self, reg):
        idx = self.registers.index(reg)
        self.register_occupied[idx] = True
        
    def free_register(self, reg):
        idx = self.registers.index(reg)
        self.register_occupied[idx] = False

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

    def compile(self, mem):
        pass

class Main(Node):
    def __init__(self):
        super().__init__()

    def compile(self, mem):
        self.children[0].compile(mem)

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

    def compile(self, mem):
        ident = self.children[0].identifier
        reg = self.children[1].compile(mem)
        mem.register_table[ident] = reg
        if ident in mem.stack_table:
            print(f"sw {reg}, {mem.stack_table[ident]}($fp)")
        else:
            mem.store_ident_in_memory(ident)

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

    def compile(self, mem):
        results = []
        for node in self.children:
            results.append(node.compile(mem))

        type = results[1]

        free_reg = mem.get_free_register()
        if type == "+":
            print(f"add {free_reg}, {results[0]}, {results[2]}")
        elif type == '-':
            print(f"sub {free_reg}, {results[0]}, {results[2]}")
        elif type == '*':
            print(f"mul {free_reg}, {results[0]}, {results[2]}")
        elif type == '/':
            print(f"div {free_reg}, {results[0]}, {results[2]}")

        return free_reg


class Operator(Node):
    def __init__(self):
        super().__init__()
        self.type = None
        self.terminal = True

    def initialize(self, value):
        self.type = value

    def exec(self, symbol_table):
        return self.type

    def compile(self, mem):
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

    def compile(self, mem):
        print("Compile method of Value")
        pass

class Var(Node):
    def __init__(self):
        super().__init__()
        self.identifier = None
        self.terminal = True

    def initialize(self, value):
        self.identifier = value

    def exec_self(self, symbol_table):
        return symbol_table[self.identifier]

    def compile(self, mem):
        return mem.load_symbol_in_free_register(self.identifier)

class Number(Node):
    def __init__(self):
        super().__init__()
        self.num = None
        self.terminal = True

    def initialize(self, value):
        self.num = int(value)

    def exec_self(self, symbol_table):
        return self.num

    def compile(self, mem):
        reg = mem.get_free_register()
        print(f"li {reg}, {self.num}")
        return reg

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

    def compile(self, mem):
        reg = self.children[0].compile(mem)

        print(f"li $v0, 1")
        print(f"move $a0, {reg}")
        print("syscall")

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

    def compile(self, mem):
        for child in self.children:
            child.compile(mem)



# 0 = var, 1 = number, 2 = if, 3 = print, 4 = equal, 5 = binary operator, 6 = open_brace, 7 = closed_brace, 8 = open_curly,
# 9 = closed_curly, 10 = semicolon
# -1 = main, -2 = statement, -3 = assignment, -4 = expression, -5 = print statement, -6 = operator expression, -7 = if statement
# -8 = block
regexes = ["^[a-zA-Z][0-9a-zA-Z]*$", "^[0-9][0-9]*$", "^if$", "^print$", "^=$", "^(\+|-|\*|/|==|!=)$",
           "^\($", "^\)$", "^\{$", "^\}$", "^;$"]

terminals = list(range(len(regexes))) #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

production_rules = {-1:[[-8]], -2:[[-3], [-2, 10, -2], [-5], [-7]], -3:[[0, 4, -4]], -4:[[1], [0], [-6], [6, -4, 7]],
                    -5:[[3, 6, -4, 7]], -6:[[-4, 5, -4]], -7:[[2, -4, 10, -8]], -8:[[8, -2, 9]]}

terminals_late_init = [0, 1, 5]


temp = [list(val) for val in production_rules.values()]
production_rules_nums = []
for lst in temp:
    production_rules_nums.extend(lst)

# {a=2;b=4;c=5;print(c);print(a);print(b)} this input crumbles bfs parsing

# check out if300 - this is an identifier in C, so it's okay
# but if+ is def not an identifier and it actually is caught only after
# compiling so lexically it's okay, which is what i expected and how this
# lexical analysis built here works

def tokenize(inp):
    final_tokens = []

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
                final_tokens.append((last_success_token, inp[:last_success_idx + 1]))
            else:
                final_tokens.append((last_success_token, ))

            print(f"{last_success_token}, {regexes[last_success_token]}")
            inp = inp[last_success_idx + 1:]
        else:
            print(f'ERROR: Invalid token "{inp}"')
            exit()

    return final_tokens

def get_production_seq_bfs(final_tokens):
    queue = [([-1], [])]
    while len(queue) > 0:
        item, rules = queue.pop(0)

        if item == final_tokens:
            return rules

        leftmost_nonterminal = -99
        leftmost_nonterminal_idx = -1
        for idx, a in enumerate(item):
            if a not in terminals:
                leftmost_nonterminal = a
                leftmost_nonterminal_idx = idx
                break

        if leftmost_nonterminal_idx == -1:
            continue

        results = production_rules[leftmost_nonterminal]

        for res in results:
            left = item[:leftmost_nonterminal_idx]
            right = item[leftmost_nonterminal_idx + 1 :]

            to_append = left + res + right

            if len(to_append) <= len(final_tokens):
                ind = production_rules_nums.index(res)
                queue.append((to_append, rules + [ind]))

    return None

# DFS is much faster than BFS
def get_production_seq_dfs(final_tokens):
    queue = [([-1], [])]
    while len(queue) > 0:
        item, rules = queue.pop()

        if item == final_tokens:
            return rules

        leftmost_nonterminal = -99
        leftmost_nonterminal_idx = -1
        for idx, a in enumerate(item):
            if a not in terminals:
                leftmost_nonterminal = a
                leftmost_nonterminal_idx = idx
                break

        if leftmost_nonterminal_idx == -1:
            continue

        results = production_rules[leftmost_nonterminal]

        for res in results:
            left = item[:leftmost_nonterminal_idx]
            right = item[leftmost_nonterminal_idx + 1 :]

            to_append = left + res + right

            if len(to_append) <= len(final_tokens):
                ind = production_rules_nums.index(res)
                queue.append((to_append, rules + [ind]))

    return None

def find_leftmost_nonterminal(tree_root):
    if not tree_root.terminal and len(tree_root.children) == 0:
        return tree_root

    for child in tree_root.children:
        res = find_leftmost_nonterminal(child)
        if res != None:
            return res

    return None


def build_parse_tree(prod_rule_seq, final_tokens):
    start = [-1]
    tree = Main()
    for rule in prod_rule_seq:
        leftmost_nonterminal_node = find_leftmost_nonterminal(tree)

        res = production_rules_nums[rule]

        for token in res:
            if token == -1:
                leftmost_nonterminal_node.add_child(Main())
            elif token == -2:
                leftmost_nonterminal_node.add_child(Statement())
            elif token == -3:
                leftmost_nonterminal_node.add_child(Assignment())
            elif token == -4:
                leftmost_nonterminal_node.add_child(Expression())
            elif token == 0:
                leftmost_nonterminal_node.add_child(Var())
            elif token == 1:
                leftmost_nonterminal_node.add_child(Number())
            elif token == -5:
                leftmost_nonterminal_node.add_child(Print())
            elif token == -6:
                leftmost_nonterminal_node.add_child(OperatorExpression())
            elif token == -7:
                leftmost_nonterminal_node.add_child(If_Statement())
            elif token == -8:
                leftmost_nonterminal_node.add_child(Block())
            elif token == 5:
                leftmost_nonterminal_node.add_child(Operator())

    list_terminals = return_list_terminals(tree)
    late_init_tokens = [token for token in final_tokens if token[0] in terminals_late_init]

    for node, token in zip(list_terminals, late_init_tokens):
        node.initialize(token[1])

    return tree

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

def return_list_terminals(tree_root):
    lst = []

    if not tree_root.terminal:
        for child in tree_root.children:
            lst.extend(return_list_terminals(child))
        return lst
    else:
        return [tree_root]

def execute(tree_root):
    symbol_table = {}

    tree_root.exec(symbol_table)

def compile(tree_root):
    mem = MemoryManager()
    tree_root.compile(mem)

if __name__ == "__main__":

    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as file:
            inp = file.read().replace('\n', '')
            print (inp)
    else:
        inp = input()

    tokens = tokenize(inp)
    tokens_nums = [token[0] for token in tokens]
    seq = get_production_seq_dfs(tokens_nums)

    if seq == None:
        print(f"ERROR: Invalid program!")
    else:
        parse_tree = build_parse_tree(seq, tokens)
        ast_tree = reduce_parse_tree_to_ast(parse_tree)
        print(seq)
        #execute(ast_tree)
        compile(ast_tree)