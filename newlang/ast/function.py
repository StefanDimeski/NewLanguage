from newlang.ast.node import *
from newlang.utils  import *
from newlang.memory_manager import *
from newlang.ar import *

class Function(Node):
    def __init__(self, name, params, body):
        super().__init__()
        self.name = name
        self.params = params
        self.body = body
        self.func_lbl = None

    def exec(self, ar):
        pass

    def compile(self, ar):
        self.func_lbl = next(MemoryManager.label_gen)
        end_function_lbl = next(MemoryManager.label_gen)

        printc(f"{self.func_lbl}:")

        ar = AR(ar)
        ar.init()

        for idx, param in enumerate(self.params):
            ar.mem.vars[param.identifier] = Var(param.identifier)
            ar.mem.vars[param.identifier].stack = 4 + idx*4

        self.body.compile(ar, end_function_lbl=end_function_lbl)

        printc(f"{end_function_lbl}:")

        ar = ar.destroy()

        printc("jr $ra")