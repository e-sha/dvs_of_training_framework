import importlib.util
import inspect
import logging


def filter_kwargs(func, kwargs):
    signature = inspect.signature(func)
    keys2use = []
    for key in signature.parameters:
        # if **kwargs is among parameters than we shouldn't filter anything
        if signature.parameters[key].kind == inspect.Parameter.VAR_KEYWORD:
            return kwargs
        if key in kwargs:
            keys2use.append(key)
    keys_not2use = [k for k in kwargs if not k in signature.parameters]
    if len(keys_not2use):
        logging.warning(f'{keys_not2use} are filtered out from OpticalFlow parameters!')
    return {key: kwargs[key] for key in keys2use}

def import_module(module_name, module_path):
    module_spec = importlib.util.find_spec(module_name, module_path)
    assert module_spec is not None, f'Module: {module_name} at {Path(module_path).resolve()} not found'
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module
