from geometry import SE2_from_xytheta, SE2value
from zuper_commons.types import ZValueError

from dg_commons import X


def ms2kmh(val: float) -> float:
    return val * 3.6


def ms2mph(val: float) -> float:
    return val * 2.23694


def kmh2ms(val: float) -> float:
    return val / 3.6


G = 9.81
""" Gravity [m/s^2]"""

rho = 1.249512
"""air density [kg/m^3]"""


def extract_pose_from_state(state: X) -> SE2value:
    try:
        pose = SE2_from_xytheta([state.x, state.y, state.theta])
        return pose
    except Exception:
        msg = "Unable to extract pose from state"
        raise ZValueError(msg=msg, state=state, state_type=type(state))


def extract_vel_from_state(state: X) -> SE2value:
    try:
        vel = state.vx
        return vel
    except Exception:
        msg = "Unable to extract vel from state"
        raise ZValueError(msg=msg, state=state, state_type=type(state))
