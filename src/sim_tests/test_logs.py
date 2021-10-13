from numpy.testing import assert_raises
from zuper_commons.types import ZValueError

from dg_commons import PlayerName
from sim import SimLog


def test_illegal_simulationLog():
    log = SimLog()

    def tryillegal():
        log[0] = {PlayerName("P1"): 1, PlayerName("P1"): 2}

    assert_raises(ZValueError, tryillegal)
