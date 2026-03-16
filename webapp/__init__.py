"""
Queering NLP Bias! - Experiments and Intuitions

"""

# Auto-export all public symbols from submodules
import importlib
import pkgutil

for _loader, _name, _is_pkg in pkgutil.iter_modules(__path__):
    if _name.startswith("_"):
        continue
    _module = importlib.import_module(f".{_name}", __name__)
    for _attr in getattr(
        _module, "__all__", [k for k in dir(_module) if not k.startswith("_")]
    ):
        globals()[_attr] = getattr(_module, _attr)
