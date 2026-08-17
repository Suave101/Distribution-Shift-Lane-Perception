# sitecustomize.py
# Workaround for Nuitka + jaxlib MLIR BlockArgumentList iteration bug

try:
    from jaxlib.mlir import ir as _ir

    def _make_iter():
        def _my_iter(self):
            for i in range(len(self)):
                yield self[i]
        return _my_iter

    # Patch any MLIR list class missing __iter__ (including BlockArgumentList)
    for attr_name in dir(_ir):
        attr = getattr(_ir, attr_name)
        if isinstance(attr, type):
            if hasattr(attr, "__len__") and hasattr(attr, "__getitem__") and not hasattr(attr, "__iter__"):
                attr.__iter__ = _make_iter()
except Exception:
    pass
