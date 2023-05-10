class MemoryTree():
    def __init__(self, parent=None):
        self.parent_memory = parent
        self.mem = {}

    def __getitem__(self, key):
        if key not in self.mem:
            if self.parent_memory is not None:
                return self.parent_memory[key]
            else:
                print(f"ERROR: Uninitialized variable '{key}'")
                exit()

        return self.mem[key]

    def __setitem__(self, key, value):
        if key in self.mem:
            self.mem[key] = value
        else:
            if self.parent_memory is not None:
                if not self.parent_memory.set_var(key, value):
                    self.mem[key] = value
            else:
                self.mem[key] = value

    def set_var(self, key, val):
        """THIS FUNCTION ONLY TO BE USED INTERNALLY"""
        if key in self.mem:
            self.mem[key] = val
            return True

        if self.parent_memory is not None:
            return self.parent_memory.set_var(key, val)

        return False

    def __delitem__(self, key):
        del self.mem[key]