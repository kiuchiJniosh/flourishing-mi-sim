from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional


PACKAGE_DIR = Path(__file__).resolve().parent
SRC_ROOT = PACKAGE_DIR.parent
PROJECT_ROOT = SRC_ROOT.parent
PACKAGE_CONFIG_DIR = PACKAGE_DIR / "config"
LEGACY_APP_DIR = PROJECT_ROOT / "app"
LEGACY_APP_CONFIG_DIR = LEGACY_APP_DIR / "config"
LEGACY_CONFIG_DIR = PROJECT_ROOT / "config"

# Backward-compatible aliases. Keep them until the staged migration is complete.
APP_DIR = PACKAGE_DIR
APP_CONFIG_DIR = PACKAGE_CONFIG_DIR


def _first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        candidate = path.expanduser()
        if candidate.exists():
            return candidate
    return None


def resolve_env_path() -> Path:
    """
    During development, search for `.env` in the project root, then legacy `app/.env`,
    then the package-local `.env`.
    """
    return _first_existing(
        (
            PROJECT_ROOT / ".env",
            LEGACY_APP_DIR / ".env",
            PACKAGE_DIR / ".env",
        )
    ) or (PROJECT_ROOT / ".env")


def resolve_config_path(filename: str) -> Path:
    """
    Prefer the package-bundled config, with legacy `app/config` and root `config`
    as migration-time fallbacks.
    """
    return _first_existing(
        (
            PACKAGE_CONFIG_DIR / filename,
            LEGACY_APP_CONFIG_DIR / filename,
            LEGACY_CONFIG_DIR / filename,
        )
    ) or (PACKAGE_CONFIG_DIR / filename)


def resolve_project_path(path: str | Path) -> Path:
    """
    Resolve relative paths in this order: package, repository root, legacy `app/`,
    and finally the current working directory.
    """
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return _first_existing(
        (
            PACKAGE_DIR / candidate,
            PROJECT_ROOT / candidate,
            LEGACY_APP_DIR / candidate,
            Path.cwd() / candidate,
        )
    ) or (PROJECT_ROOT / candidate)
