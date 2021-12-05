from dg_commons.sim import logger
from dg_commons.sim.models.model_structures import ModelParameters
from dg_commons.sim.models.spacecraft_structures import SpacecraftParameters


def apply_speed_constraint(speed: float, acceleration: float, p: ModelParameters):
    """Enforces acceleration limits if the maximum speed is reached"""
    if (speed <= p.vx_limits[0] and acceleration < 0) or (speed >= p.vx_limits[1] and acceleration > 0):
        acceleration = 0
        logger.warn(
            f"Reached min or max velocity, acceleration set to {acceleration:.2f}: \nspeed {speed:.2f}\tspeed limits [{p.vx_limits[0]:.2f},{p.vx_limits[1]:.2f}]"
        )
    return acceleration


def apply_acceleration_limits(acceleration: float, p: ModelParameters) -> float:
    """Enforces actuation limits for acceleration"""
    if acceleration < p.acc_limits[0]:
        logger.warn(f"Commanded acceleration below limits, clipping value: {acceleration:.2f}<{p.acc_limits[0]:.2f}")
        acceleration = p.acc_limits[0]
    elif acceleration > p.acc_limits[1]:
        logger.warn(f"Commanded acceleration above limits, clipping value: {acceleration:.2f}>{p.acc_limits[1]:.2f}")
        acceleration = p.acc_limits[1]
    return acceleration


def apply_full_acceleration_limits(speed: float, acceleration: float, p: ModelParameters):
    """Enforces acceleration limits if the maximum speed is reached"""
    acc = apply_acceleration_limits(acceleration, p)
    acc = apply_speed_constraint(speed, acc, p)
    return acc


# todo make it more generic also for the other models
def apply_rot_speed_constraint(omega: float, domega: float, p: SpacecraftParameters):
    """Enforces rot acceleration limits if the maximum speed is reached"""
    if (omega <= p.dpsi_limits[0] and domega < 0) or (omega >= p.dpsi_limits[1] and domega > 0):
        domega = 0
        logger.warn(
            f"Reached min or max rotation speed, acceleration set to {domega:.2f}: \nspeed {omega:.2f}\tspeed limits [{p.dpsi_limits[0]:.2f},{p.dpsi_limits[1]:.2f}]"
        )
    return domega
