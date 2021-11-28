__version__ = "0.0.10"

from logging import INFO

import contracts
from zuper_commons.logs import ZLogger

logger = ZLogger(__name__)

logger.setLevel(level=INFO)

contracts.disable_all()

from .seq import *
from .geo import *
from .game_types import *
from .utils_types import *
from .utils_toolz import *
