from __future__ import annotations

from pathlib import Path


def resolve_with_base_dir(base_dir: str, user_path: str) -> str:
    """
    Resolve a user-provided path against a base directory.

    - Empty input returns empty string.
    - Absolute paths are returned as-is (after stripping).
    - Relative paths are joined to base_dir (base_dir is expanded with ~).
    """
    if user_path is None:
        return ""

    user_path = str(user_path).strip()
    if not user_path:
        return ""

    p = Path(user_path)
    if p.is_absolute():
        return user_path

    base = Path(str(base_dir).strip() if base_dir is not None else "").expanduser()
    if not str(base):
        # no base_dir provided: keep relative path
        return user_path

    return str(base / p)

