from argparse import Namespace

# Global config imported elsewhere.

class Dict(dict):
    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value

# opt = Namespace()
opt = Dict()
data = {}
w2v = {}
loaders = {}
global_logger = {}