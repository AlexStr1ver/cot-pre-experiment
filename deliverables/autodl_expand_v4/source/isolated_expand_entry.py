#!/usr/bin/env python3
"""Launch known package entry points under Python -I, without cwd/PYTHONPATH imports."""
import importlib
from pathlib import Path
import runpy
import sys


def main():
    if not sys.flags.isolated:
        raise SystemExit('Use python -I isolated_expand_entry.py ...')
    if len(sys.argv) < 3:
        raise SystemExit('Expected module unittest ... or script expanded_experiment.py/status_expanded.py ...')
    kind, target, *arguments = sys.argv[1:]
    allowed = (kind == 'module' and target == 'unittest') or (
        kind == 'script' and target in ('expanded_experiment.py', 'collect_phase2.py', 'prepare_expansion_evidence.py', 'status_expanded.py'))
    if not allowed:
        raise SystemExit('Unsupported entry point')
    package = Path(__file__).resolve().parent
    # Import installed libraries BEFORE adding our sibling modules. A stray
    # torch.py/transformers directory in a working directory cannot replace them.
    if target != 'status_expanded.py':
        importlib.import_module('torch')
        importlib.import_module('transformers')
    sys.path.insert(0, str(package))
    sys.argv = [target, *arguments]
    if kind == 'module':
        runpy.run_module(target, run_name='__main__', alter_sys=True)
    else:
        runpy.run_path(str(package / target), run_name='__main__')


if __name__ == '__main__':
    main()
