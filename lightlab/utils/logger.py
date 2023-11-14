import os
import sys
from typing import Any
import termcolor
import logging
from datetime import datetime

if os.name == "nt":
    import colorama

    colorama.init()

COLORS = {
    "WARNING": "yellow",
    "INFO": "white",
    "DEBUG": "blue",
    "CRITICAL": "red",
    "ERROR": "red",
}


class ColoredFormatter(logging.Formatter):
    def __init__(self, fmt, use_color=True) -> Any:
        logging.Formatter.__init__(self, fmt)
        self.use_color = use_color

    def format(self, record):
        levelname = record.levelname

        if self.use_color and levelname in COLORS:

            def colored(text):
                return termcolor.colored(
                    text,
                    color=COLORS[levelname],
                    attrs={"bold": True},
                )

            record.levelname2 = colored("{:<7}".format(record.levelname))
            record.message2 = colored(record.msg)

            asctime2 = datetime.fromtimestamp(record.created)
            record.asctime2 = termcolor.colored(asctime2, color="green")

            record.module2 = termcolor.colored(record.module, color="cyan")
            record.funcName2 = termcolor.colored(record.funcName, color="cyan")
            record.lineno2 = termcolor.colored(record.lineno, color="cyan")

        return logging.Formatter.format(self, record)


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler(sys.stderr)
handler_format = ColoredFormatter(
    "%(asctime2)s [%(levelname2)s] %(module2)s:%(funcName2)s:%(lineno2)s - %(message2)s"
)
stream_handler.setFormatter(handler_format)
LOGGER.addHandler(stream_handler)

if __name__ == "__main__":
    LOGGER.error("hello")
    LOGGER.info("hello")
    LOGGER.warning("hello")
    LOGGER.critical("hello")
    LOGGER.debug("hello")
