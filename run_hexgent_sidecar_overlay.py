import pathlib
import sys
import sglang
local_sglang = str(pathlib.Path('/workspace/sglang_hexgent_debug/python/sglang'))
if local_sglang not in sglang.__path__:
    sglang.__path__.insert(0, local_sglang)
from sglang.hexgent.sidecar import main
main()
