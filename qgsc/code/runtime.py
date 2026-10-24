
from pathlib import Path
from types import SimpleNamespace
import copy
import sys
import threading

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ISING_ROOT = PACKAGE_ROOT.parent
for _path in (str(PACKAGE_ROOT), str(PACKAGE_ROOT / "core")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

settings = SimpleNamespace()
run_lock = threading.Lock()


def configure(parameters):
    settings.__dict__.clear()
    settings.__dict__.update(copy.deepcopy(parameters))
