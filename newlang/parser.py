from newlang.ttype import *
from newlang.ast.program import *
from newlang.ast.variable import *
from newlang.ast.function import *
from newlang.ast.function_call import *
from newlang.ast.block import *
from newlang.ast.lang_print import *
from newlang.ast.lang_operator import *
from newlang.ast.number import *
from newlang.ast.lang_input import *

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

        # This is done this way so that recursive calls in body can find where the name points to
        node = Function(ident, params, None)
        self.func_name_to_ast_node[ident] = node

        body = self.parse_block()
        node.body = body


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
        elif nxt.type == TType.WHILE:
            return self.parse_while_statement()
        elif nxt.type == TType.IDENT:
            if self.lex.peek(2).type == TType.OPEN_BRACE:
                return self.parse_function_call()
            else:
                return self.parse_assignment()
        elif nxt.type == TType.BREAK:
            return self.parse_break()
        elif nxt.type == TType.CONTINUE:
            return self.parse_continue()
        elif nxt.type == TType.RETURN:
            return self.parse_return()
        else:
            print(f"ERROR: Unexpected token {nxt.val}")
            exit()

    def parse_break(self):
        lst = []
        lst.append(self.lex.eat(TType.BREAK))

        while self.lex.peek().type == TType.BREAK:
            lst.append(self.lex.eat(TType.BREAK))

        if self.lex.peek().type == TType.CONTINUE:
            lst.append(self.lex.eat(TType.CONTINUE))

        if self.lex.peek().type == TType.BREAK:
            print(f"ERROR: Invalid 'break' after 'continue'")
            exit()

        return Break(lst)

    def parse_continue(self):
        self.lex.eat(TType.CONTINUE)
        return Continue()

    def parse_return(self):
        self.lex.eat(TType.RETURN)
        to_ret = self.parse_expr1()

        return Return(to_ret)

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
        start_is_lower = False

        self.lex.eat(TType.FOR)
        lower = self.parse_expr1()

        if self.lex.peek().type == TType.IDENT and self.lex.peek().val == 's':
            self.lex.eat(TType.IDENT)
            start_is_lower = True

        a = self.lex.eat(TType.BIN_OP_GREATER_SMALLER)
        if a.val not in ["<", "<="]:
            print(f"ERROR: Invalid syntax. Unexpected token {a.val}")
            exit()

        # var_name = self.lex.eat(TType.IDENT)
        # up_down = self.lex.eat(TType.BIN_OP_PLUS_MIN)

        # Im envisioning it to be something like for 0s <= a+50 <= 200 { block }
        # where s indicates the which of the bounds is the starting value

        var_name = self.lex.peek()
        var_expr = self.parse_expr1()

        b = self.lex.eat(TType.BIN_OP_GREATER_SMALLER)
        if b.val not in ["<", "<="]:
            print(f"ERROR: Invalid syntax. Unexpected token {b.val}")
            exit()

        upper = self.parse_expr1()

        if self.lex.peek().type == TType.IDENT and self.lex.peek().val == 's':
            if start_is_lower:
                print(f"ERROR: The modifier 's' must be included either on the lower or on the upper bound, but not both")
                exit()

            self.lex.eat(TType.IDENT)
        elif not start_is_lower:
            print(f"ERROR: For loop start bound not set. Please use 's' after one of the bound expressions to indicate that the for loop starts there")
            exit()

        internal_block = self.parse_block()

        return For_Stat(lower, upper, internal_block, var_name.val, var_expr, start_is_lower, a.val, b.val)

    def parse_while_statement(self):
        self.lex.eat(TType.WHILE)
        self.lex.eat(TType.OPEN_BRACE)
        cond = self.parse_expr()
        self.lex.eat(TType.CLOSED_BRACE)

        internal_block = self.parse_block()

        return While_Stat(cond, internal_block)


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
            if self.lex.peek(2).type == TType.OPEN_BRACE:
                node = self.parse_function_call()
            else:
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