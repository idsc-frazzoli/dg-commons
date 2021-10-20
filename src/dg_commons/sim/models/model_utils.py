from dg_commons.sim import logger
from dg_commons.sim.models.model_structures import ModelParameters


def acceleration_constraint(speed: float, acceleration: float, p: ModelParameters):
    """Enforces acceleration limits"""
    if (speed <= p.vx_limits[0] and acceleration < 0) or (speed >= p.vx_limits[1] and acceleration > 0):
        acceleration = 0
        logger.warn(
            f"Reached min or max velocity, acceleration set to {acceleration:.2f}: \nspeed {speed:.2f}\tspeed limits [{p.vx_limits[0]:.2f},{p.vx_limits[1]:.2f}]"
        )
    elif acceleration < p.acc_limits[0]:
        logger.warn(f"Commanded acceleration below limits, clipping value: {acceleration:.2f}<{p.acc_limits[0]:.2f}")
        acceleration = p.acc_limits[0]
    elif acceleration > p.acc_limits[1]:
        logger.warn(f"Commanded acceleration above limits, clipping value: {acceleration:.2f}>{p.acc_limits[1]:.2f}")
        acceleration = p.acc_limits[1]
    return acceleration
