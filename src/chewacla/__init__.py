"""chewacla package metadata and version loader."""

__settings_orgName__ = "prjemian"
__package_name__ = "chewacla"

import importlib.metadata

def _get_version(version_module=None):
    """Return package version.

    Priority:
    1. If a generated _version.py exists (written by setuptools_scm), use it.
    2. Try importlib.metadata.version().
    3. Fall back to a safe default.
    """
    # 1) try generated version file (recommended)
    if version_module is not None:
        ver = getattr(version_module, "version", None)
        if ver:
            return ver

    # 2) try importlib.metadata
    try:
        return importlib.metadata.version(__package_name__)
    except Exception:
        pass

    # 3) fallback
    return "0+unknown"

__version__ = _get_version()

# For testing purposes, we can expose the _get_version function
if __name__ == "__main__":
    print(f"Package version: {__version__}")
