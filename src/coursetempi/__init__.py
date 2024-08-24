from coursetempi._version import __version__


def get_version() -> str:
    """
    Get the version of the package. This is the cli function of the package.
    If you want to check the version of the package you can use this function.


    Returns:
        The version of the package as a string e.g. "0.1.0"
    """
    return __version__
