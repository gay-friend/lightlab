from importlib import metadata
import re
import torch


from .logger import LOGGER


def is_ascii(s) -> bool:
    # Convert list, tuple, None, etc. to string
    s = str(s)

    # Check if the string is composed of only ASCII characters
    return all(ord(c) < 128 for c in s)


def parse_version(version="0.0.0") -> tuple:
    try:
        return tuple(
            map(int, re.findall(r"\d+", version)[:3])
        )  # '2.0.1+cpu' -> (2, 0, 1)
    except Exception as e:
        LOGGER.warning(
            f"WARNING ⚠️ failure for parse_version({version}), returning (0, 0, 0): {e}"
        )
        return 0, 0, 0


def check_version(
    current: str = "0.0.0",
    required: str = "0.0.0",
    name: str = "version",
    verbose: bool = False,
) -> bool:
    if not current:  # if current is '' or None
        LOGGER.warning(
            f"WARNING ⚠️ invalid check_version({current}, {required}) requested, please check values."
        )
        return True
    elif not current[0].isdigit():
        try:
            name = current  # assigned package name to 'name' arg
            current = metadata.version(current)  # get version string from package name
        except metadata.PackageNotFoundError:
            return False

    if not required:  # if required is '' or None
        return True

    result = True
    c = parse_version(current)  # '1.2.3' -> (1, 2, 3)
    for r in required.strip(",").split(","):
        op, v = re.match(
            r"([^0-9]*)([\d.]+)", r
        ).groups()  # split '>=22.04' -> ('>=', '22.04')
        v = parse_version(v)  # '1.2.3' -> (1, 2, 3)
        if op == "==" and c != v:
            result = False
        elif op == "!=" and c == v:
            result = False
        elif op in (">=", "") and not (
            c >= v
        ):  # if no constraint passed assume '>=required'
            result = False
        elif op == "<=" and not (c <= v):
            result = False
        elif op == ">" and not (c > v):
            result = False
        elif op == "<" and not (c < v):
            result = False
    if not result:
        warning_message = f"WARNING ⚠️ {name}{op}{required} is required, but {name}=={current} is currently installed"
        if verbose:
            LOGGER.warning(warning_message)
    return result
