import py_compile, pathlib
errs=[]
for p in pathlib.Path('backend').rglob('*.py'):
    try:
        py_compile.compile(str(p), doraise=True)
    except Exception as e:
        print('ERROR', p, e)
        errs.append((p, e))
print('Done', len(errs), 'errors')
