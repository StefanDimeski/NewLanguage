from newlang.ast.node import *
from newlang.utils import *
from newlang.flag import *
from newlang.memory_manager import *
from newlang.memory_tree import *
from newlang.ast.if_statement import *
from newlang.ast.lang_return import *
from newlang.ast.lang_continue import *
from newlang.ast.lang_break import *
from newlang.ast.while_stat import *
from newlang.ast.for_stat import *

class Block(Node):
    def __init__(self, children):
        super().__init__()
        self.children = children

    def exec(self, ar, create_new_memory=True):
        if create_new_memory:
            new_mem = MemoryTree(ar.memory)
            ar.memory = new_mem

        for child in self.children:
            ret = child.exec(ar)

            if ret is not None:
                if create_new_memory:
                    ar.memory = ar.memory.parent_memory
                return ret


        # Return it back (i.e. discard variables that were created in the scope of the block that just ended)
        if create_new_memory:
            ar.memory = ar.memory.parent_memory

        return None

    def compile(self, ar, end_function_lbl=None, create_new_mem=True, loop_continue_lbls=None, loop_break_lbls=None):
        if loop_break_lbls is None:
            loop_break_lbls = []
        if loop_continue_lbls is None:
            loop_continue_lbls = []

        mem = ar.mem

        if create_new_mem:
            mem = MemoryManager.create_new_mem(mem)

        for child in self.children:
            if isinstance(child, Return):
                child.compile(ar, end_function_lbl=end_function_lbl)
            elif isinstance(child, If_Statement) or isinstance(child, While_Stat) or isinstance(child, For_Stat):
                child.compile(ar, end_function_lbl=end_function_lbl, loop_continue_lbls=loop_continue_lbls, loop_break_lbls=loop_break_lbls)
            elif isinstance(child, Break) or isinstance(child, Continue):
                child.compile(ar, loop_continue_lbls=loop_continue_lbls, loop_break_lbls=loop_break_lbls)
            else:
                child.compile(ar)

        if create_new_mem:
            # This will update mem accordingly
            mem = MemoryManager.go_back_a_memory(mem)