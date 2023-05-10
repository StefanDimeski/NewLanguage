#import pyperclip

from newlang.memory_tree import *
from newlang.activation_record import *

full_str = ""
def printc(str):
    global full_str
    full_str += (f"{str}\n")
    print(str)

def label_generator():
    num = 0
    while True:
        num_str = str(num)
        label = ""

        for digit in num_str:
            label += chr(97 + int(digit))

        num += 1
        yield label

def varname_generator():
    num = 0
    while True:
        num_str = str(num)
        label = "__"

        for digit in num_str:
            label += chr(97 + int(digit))

        num += 1
        yield label

def execute(tree_root):
    mem = MemoryTree()
    ar_main = ActivationRecord(mem, None)

    tree_root.exec(ar_main)

def compile(ast):
    global full_str
    ast.compile()
    #pyperclip.copy(full_str)