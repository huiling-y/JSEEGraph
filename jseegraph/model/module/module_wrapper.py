class ModuleWrapper:
    def __init__(self, module):
        self.module = module

    def __call__(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def parameters(self):
        return self.module.parameters()