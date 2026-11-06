"""Optional process-local MSVC preload; never installs or changes system libraries."""

import ctypes
import hashlib
import os
from pathlib import Path

_HANDLES = []


def prepare_runtime():
    if os.name != "nt":
        return {"status": "not-needed"}
    override = os.environ.get("HNET_MSVC_DLL")
    path = Path(override) if override else Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin/msvcp140.dll"
    if not path.is_file():
        return {"status": "not-preloaded", "note": "Install official MSVC runtime or set HNET_MSVC_DLL to an existing trusted x64 runtime."}
    _HANDLES.append(ctypes.WinDLL(str(path.resolve())))
    return {"status": "process-local-preload", "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "note": "Reuses existing runtime; not copied, redistributed or installed system-wide."}


RUNTIME = prepare_runtime()
