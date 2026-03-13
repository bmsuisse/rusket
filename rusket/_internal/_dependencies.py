import types


def import_optional_dependency(
    name: str,
    extra: str = "",
    errors: str = "raise",
) -> types.ModuleType:
    """
    Import an optional dependency.

    By default, if a dependency is missing an ImportError with a nice
    message will be raised.

    Parameters
    ----------
    name : str
        The module name.
    extra : str
        Additional text to include in the ImportError message.
    errors : str {'raise', 'warn', 'ignore'}
        What to do when a dependency is not found:
        - raise : Raise an ImportError.
        - warn : Warn that the dependency is missing.
        - ignore : Return None.

    Returns
    -------
    module
    """

    import importlib.util

    package_name = name.split(".")[0]
    install_name = package_name

    # Mapping of module name to pip install name if they differ
    install_mapping = {
        "pyspark": "pyspark",
        "polars": "polars",
        "pandas": "pandas",
        "optuna": "optuna",
        "mlflow": "mlflow",
        "pyarrow": "pyarrow",
    }
    install_name = install_mapping.get(package_name, package_name)

    spec = importlib.util.find_spec(name)
    if spec is None:
        msg = f"Missing optional dependency '{package_name}'. Use pip or conda to install {install_name}."
        if extra:
            msg += f" {extra}"

        if errors == "raise":
            raise ImportError(msg)
        elif errors == "warn":
            import warnings

            warnings.warn(msg, UserWarning, stacklevel=2)
            return None  # type: ignore
        elif errors == "ignore":
            return None  # type: ignore
        else:
            raise ValueError(f"Invalid value for errors: {errors}")

    module = importlib.import_module(name)
    return module
