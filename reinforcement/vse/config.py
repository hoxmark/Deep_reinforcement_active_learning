
# Global config imported and set elsewhere.

class Dict(dict):
    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value

opt = Dict()
data = Dict()
w2v = {}
loaders = {}
global_logger = {}
