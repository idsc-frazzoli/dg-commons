from typing import Tuple

from commonroad.geometry.shape import Rectangle as crRectangle
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType

from dg_commons import Color
from dg_commons.sim import SimModel
from dg_commons.sim.models import Pacejka
from dg_commons.sim.models.pedestrian import PedestrianModel, PedestrianState
from dg_commons.sim.models.vehicle_dynamic import VehicleModelDyn, VehicleStateDyn
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from dg_commons.sim.models.vehicle_utils import VehicleParameters
from dg_commons.sim.scenarios import NotSupportedConversion

__all__ = ["infer_model_from_cr_dyn_obstacle"]


def infer_model_from_cr_dyn_obstacle(dyn_obs: DynamicObstacle, color: Color) -> SimModel:
    """Recover a simulation model form a Commonroad dynamic obstacle"""
    if dyn_obs.obstacle_type in [ObstacleType.CAR, ObstacleType.BICYCLE, ObstacleType.TRUCK, ObstacleType.BUS]:
        axle_length_ratio = 0.7  # the distance between wheels is less than the car body
        axle_width_ratio = 0.95  # the distance between wheels is less than the car body

        assert isinstance(dyn_obs.obstacle_shape, crRectangle)
        l = dyn_obs.obstacle_shape.length * axle_length_ratio
        dtheta = dyn_obs.prediction.trajectory.state_list[0].orientation - dyn_obs.initial_state.orientation
        delta = dtheta / l
        x0 = VehicleStateDyn(
            x=dyn_obs.initial_state.position[0],
            y=dyn_obs.initial_state.position[1],
            psi=dyn_obs.initial_state.orientation,
            vx=dyn_obs.initial_state.velocity,
            delta=delta,
        )
        w_half = dyn_obs.obstacle_shape.width / 2 * axle_width_ratio

        if dyn_obs.obstacle_type == ObstacleType.BICYCLE:
            vg = VehicleGeometry.default_bicycle(
                w_half=w_half,
                lf=l / 2.0,
                lr=l / 2.0,
                color=color,
            )
            vp = VehicleParameters.default_bicycle()
        else:
            mass, rot_inertia = _estimate_car_mass_inertia(
                length=dyn_obs.obstacle_shape.length, width=dyn_obs.obstacle_shape.width
            )
            vg = VehicleGeometry.default_car(
                w_half=w_half,
                m=mass,
                Iz=rot_inertia,
                lf=l / 2.0,
                lr=l / 2.0,
                color=color,
            )
            vp = VehicleParameters.default_car()
        model = VehicleModelDyn(
            x0=x0, vg=vg, vp=vp, pacejka_front=Pacejka.default_car_front(), pacejka_rear=Pacejka.default_car_rear()
        )
    elif dyn_obs.obstacle_type == ObstacleType.PEDESTRIAN:
        x0 = PedestrianState(
            x=dyn_obs.initial_state.position[0],
            y=dyn_obs.initial_state.position[1],
            psi=dyn_obs.initial_state.orientation,
            vx=dyn_obs.initial_state.velocity,
            vy=0,
            dpsi=0,
        )
        model = PedestrianModel.default(x0)
    else:
        raise NotSupportedConversion(
            "Cannot convert Commonroad type to dg_commons model", commonroad_type=dyn_obs.obstacle_type
        )
    return model


def _estimate_car_mass_inertia(length: float, width: float) -> Tuple[float, float]:
    """#todo justify and fix this empirical formulas"""
    alpha = 50
    beta = 1.6
    area = length * width
    mass = alpha * area**beta
    inertia = mass * (length + width) / 6
    return mass, inertia
