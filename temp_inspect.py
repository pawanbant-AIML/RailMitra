import importlib.util, sys, os
p = os.path.join(os.getcwd(),'backend','app','agent','query_understanding.py')
spec = importlib.util.spec_from_file_location('qu', p)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
QU = getattr(mod,'QueryUnderstanding')
inst = QU()
print('has _normalize?', hasattr(inst,'_normalize'))
print('methods sample:', [m for m in dir(inst) if m.startswith('_') and not m.startswith('__')][:40])
