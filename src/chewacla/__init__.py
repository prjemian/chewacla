"""chewacla package metadata and version loader."""

__settings_orgName__ = "prjemian"
__package_name__ = "chewacla"


def _get_version():
    """Return package version.

    Priority:
    1. If a generated _version.py exists (written by setuptools_scm), use it.
    2. Try importlib.metadata.version().
    3. Fall back to a safe default.
    """
    # 1) try generated version file (recommended)
    try:
        from . import _version  # type: ignore
    except Exception:
        _version = None

    if _version is not None:
        ver = getattr(_version, "version", None)
        if ver:
            return ver

    # 2) try importlib.metadata
    try:
        import importlib.metadata

        return importlib.metadata.version(__package_name__)

    except Exception:
        pass

    # 3) fallback
    return "0+unknown"


__version__ = _get_version()
