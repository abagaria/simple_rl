import pdb

class Predicate(object):

    def __init__(self, func, name=""):
        self.func = func
        self.name = name

    def is_true(self, x):
        return self.func(x)
