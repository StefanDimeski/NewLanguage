class Var():
    def __init__(self, ident, reg=None):
        self.reg = reg
        self.stack = None
        self.ident = ident
        self.temp = False
        self.dirty = False
        self.evictible = True