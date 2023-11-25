from dg_commons.sim import logger
from dg_commons.sim.models.model_structures import ModelParameters
from dg_commons.sim.models.spacecraft_structures import SpacecraftParameters
from dg_commons.sim.models.rocket_structures import RocketParameters


def apply_speed_constraint(speed: float, acceleration: float, p: ModelParameters):
    """Enforces acceleration limits if the maximum speed is reached"""
    if (speed <= p.vx_limits[0] and acceleration < 0) or (speed >= p.vx_limits[1] and acceleration > 0):
        acceleration = 0
        logger.debug(
            f"Reached min or max velocity, acceleration set to {acceleration:.2f}: \nspeed {speed:.2f}\tspeed limits [{p.vx_limits[0]:.2f},{p.vx_limits[1]:.2f}]"
        )
    return acceleration


def apply_acceleration_limits(acceleration: float, p: ModelParameters) -> float:
    """Enforces actuation limits for acceleration"""
    if acceleration < p.acc_limits[0]:
        logger.debug(f"Commanded acceleration below limits, clipping value: {acceleration:.2f}<{p.acc_limits[0]:.2f}")
        acceleration = p.acc_limits[0]
    elif acceleration > p.acc_limits[1]:
        logger.debug(f"Commanded acceleration above limits, clipping value: {acceleration:.2f}>{p.acc_limits[1]:.2f}")
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
        logger.debug(
            f"Reached min or max rotation speed, acceleration set to {domega:.2f}: \nspeed {omega:.2f}\tspeed limits [{p.dpsi_limits[0]:.2f},{p.dpsi_limits[1]:.2f}]"
        )
    return domega


# todo make it more generic also for the other models
def apply_force_limits(force: float, p: RocketParameters):
    """Enforces force limits"""

    if not (p.F_limits[0] < force < p.F_limits[1]):
        force1 = max(p.F_limits[0], min(force, p.F_limits[0]))
        logger.debug(
            f"Reached force limits, force set to {force1:.2f}: \n"
            f"requested force {force:.2f}\toutside limits [{p.F_limits[0]:.2f},{p.F_limits[1]:.2f}]"
        )
        force = force1
    return force


def apply_ang_vel_limits(ang_vel: float, p: RocketParameters):
    """Enforces angular velocity limits"""
    if ang_vel <= p.dphi_limits[0]:
        ang_vel = p.dphi_limits[0]
        logger.debug(
            f"Min angular velocity reached, angular velocity set to {ang_vel:.2f}: \n"
            f"angular velocity {ang_vel:.2f}\tangular velocity limits [{p.dphi_limits[0]:.2f},{p.dphi_limits[1]:.2f}]"
        )
    elif ang_vel >= p.dphi_limits[1]:
        ang_vel = p.dphi_limits[1]
        logger.debug(
            f"Reached max angular velocity, angular velocity set to {ang_vel:.2f}: \n"
            f"angular velocity {ang_vel:.2f}\tangular velocity limits [{p.dphi_limits[0]:.2f},{p.dphi_limits[1]:.2f}]"
        )
    return ang_vel


def apply_ang_constraint(ang: float, dang: float, p: RocketParameters):
    """Enforces angular acceleration limits if the maximum speed is reached"""
    if (ang <= p.phi_limits[0] and dang < 0) or (ang >= p.phi_limits[1] and dang > 0):
        dang = 0
        logger.debug(
            f"Reached min or max angle, acceleration set to {dang:.2f}: \n"
            f"angle {ang:.2f}\tangle limits [{p.phi_limits[0]:.2f},{p.phi_limits[1]:.2f}]"
        )
    return dang


def apply_full_ang_vel_limits(ang: float, dang: float, p: RocketParameters):
    """Enforces angular velocity limits if the maximum speed is reached"""
    dang = apply_ang_vel_limits(dang, p)
    dang = apply_ang_constraint(ang, dang, p)
    return dang
