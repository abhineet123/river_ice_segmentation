import os

def linux_path(*args, **kwargs):
    return os.path.join(*args, **kwargs).replace(os.sep, '/')

