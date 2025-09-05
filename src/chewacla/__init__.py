"""chewacla"""

__settings_orgName__ = "prjemian"
__package_name__ = "chewacla"

def _get_version():
    """Make the version code testable."""
    import importlib.metadata
    import importlib.util

    text = importlib.metadata.version(__package_name__)

    if importlib.util.find_spec("setuptools_scm") is not None:
        """Preferred source of package version information."""
        import setuptools_scm

        try:
            text = setuptools_scm.get_version(root="..", relative_to=__file__)
        except LookupError:
            pass  # TODO: How to test this?

    return text


__version__ = _get_version()
