from newlang.ast.node import *

class Variable(Node):
    def __init__(self, ident):
        super().__init__()
        self.identifier = ident

    def exec(self, ar):
        return ar.memory[self.identifier]

    def compile(self, ar):
        return self.identifier