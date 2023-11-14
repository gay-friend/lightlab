import torch
import os
from tqdm import tqdm as tqdm_original
from matplotlib import pyplot as plt
import contextlib
import threading

from .logger import LOGGER
from .checks import check_version

__all__ = ["LOGGER"]
TORCH_1_10 = check_version(torch.__version__, "1.10.0")
IMAGES_SUFFIX = (".jpg", ".jpeg", ".png", ".bmp")
RANK = int(os.getenv("RANK", -1))
VERBOSE = str(os.getenv("VERBOSE", True)).lower() == "true"  # global verbose mode
TQDM_BAR_FORMAT = "{l_bar}{bar:10}{r_bar}" if VERBOSE else None  # tqdm bar format


def threaded(func):
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
        thread.start()
        return thread

    return wrapper


class ThreadingLocked:
    def __init__(self):
        """Initializes the decorator class for thread-safe execution of a function or method."""
        self.lock = threading.Lock()

    def __call__(self, f):
        """Run thread-safe execution of function or method."""
        from functools import wraps

        @wraps(f)
        def decorated(*args, **kwargs):
            """Applies thread-safety to the decorated function or method."""
            with self.lock:
                return f(*args, **kwargs)

        return decorated


class SimpleClass:
    def __str__(self):
        attr = []
        for a in dir(self):
            v = getattr(self, a)
            if not callable(v) and not a.startswith("_"):
                if isinstance(v, SimpleClass):
                    # Display only the module and class name for subclasses
                    s = f"{a}: {v.__module__}.{v.__class__.__name__} object"
                else:
                    s = f"{a}: {repr(v)}"
                attr.append(s)
        return (
            f"{self.__module__}.{self.__class__.__name__} object with attributes:\n\n"
            + "\n".join(attr)
        )

    def __repr__(self):
        return self.__str__()

    def __getattr__(self, attr):
        name = self.__class__.__name__
        raise AttributeError(
            f"'{name}' object has no attribute '{attr}'. See valid attributes below.\n{self.__doc__}"
        )


class TQDM(tqdm_original):
    def __init__(self, *args, **kwargs):
        # Set new default values (these can still be overridden when calling TQDM)
        kwargs["disable"] = not VERBOSE or kwargs.get(
            "disable", False
        )  # logical 'and' with default value if passed
        kwargs.setdefault(
            "bar_format", TQDM_BAR_FORMAT
        )  # override default value if passed
        super().__init__(*args, **kwargs)


def plt_settings(rcparams=None, backend="Agg"):
    if rcparams is None:
        rcparams = {"font.size": 11}

    def decorator(func):
        """Decorator to apply temporary rc parameters and backend to a function."""

        def wrapper(*args, **kwargs):
            """Sets rc parameters and backend, calls the original function, and restores the settings."""
            original_backend = plt.get_backend()
            if backend != original_backend:
                plt.close(
                    "all"
                )  # auto-close()ing of figures upon backend switching is deprecated since 3.8
                plt.switch_backend(backend)

            with plt.rc_context(rcparams):
                result = func(*args, **kwargs)

            if backend != original_backend:
                plt.close("all")
                plt.switch_backend(original_backend)
            return result

        return wrapper

    return decorator


class TryExcept(contextlib.ContextDecorator):
    def __init__(self, msg="", verbose=True):
        self.msg = msg
        self.verbose = verbose

    def __enter__(self):
        pass

    def __exit__(self, exc_type, value, traceback):
        if self.verbose and value:
            LOGGER.warning(f"{self.msg}{': ' if self.msg else ''}{value}")
        return True
