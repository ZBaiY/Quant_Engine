from __future__ import annotations

from pathlib import Path
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)  # type: ignore[name-defined]

_repo_root = Path(__file__).resolve().parents[2]
_external_apps = _repo_root / "apps"
if _external_apps.is_dir():
    __path__.append(str(_external_apps))
