import re
from newlang.utils import *
from newlang.ttype import *
from newlang.lang_token import *

# check out if300 - this is an identifier in C, so it's okay
# but if+ is def not an identifier and it actually is caught only after
# compiling so lexically it's okay, which is what i expected and how this
# lexical analysis built here works


regexes = ["^[a-zA-Z][0-9a-zA-Z]*$", "^[0-9][0-9]*$", "^if$", "^print$", "^else$", "^for$", "^while$", "^function$", "^continue$", "^break$",
           "^return$", "^=$", "^(\+|-)$", "^(\*|/|%)$", "^(==|!=)$", "^(<|>|<=|>=)$", "^\($", "^\)$", "^\{$", "^\}$", "^;$", "^~$", "^\*\*\*$"]


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
            print(f"ERROR: Invalid token '{nxt.val}', '{token_type}' expected!")
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