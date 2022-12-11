__version__ = "0.0.25"

from logging import INFO
from typing import ClassVar

import contracts
from zuper_commons.logs import ZLogger

logger = ZLogger(__name__)

logger.setLevel(level=INFO)

contracts.disable_all()


class DgCommonsConstants:
    """Global constants for the library."""

    checks: ClassVar[bool] = True
    """
        If true activates extra safety checks and assertions.
        Mainly on object creations and methods.
    """


from .seq import *
from .geo import *
from .game_types import *
from .utils_types import *
from .utils_toolz import *
