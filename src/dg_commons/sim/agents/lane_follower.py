from typing import Optional

import numpy as np
from geometry import SE2_from_xytheta, SE2value, translation_from_SE2

from dg_commons import PlayerName, X, DgSampledSequence, SE2_apply_T2
from dg_commons.controllers.pure_pursuit import PurePursuit
from dg_commons.controllers.speed import SpeedBehavior, SpeedController
from dg_commons.controllers.steer import SteerController
from dg_commons.maps import DgLanelet
from dg_commons.planning.trajectory import Trajectory
from dg_commons.sim import SimObservations, DrawableTrajectoryType, InitSimObservations
from dg_commons.sim.agents.agent import Agent
from dg_commons.sim.models import ModelParameters, ModelGeometry
from dg_commons.sim.models.vehicle import VehicleCommands, VehicleState
from dg_commons.sim.models.vehicle_ligths import LightsCmd, LightsValues
from dg_commons.sim.models.vehicle_structures import VehicleGeometry


class LFAgent(Agent):
    """This agent is a simple lane follower tracking the centerline of the given lane
    via a pure pursuit controller. The reference in speed is determined by the speed behavior.
    """

    def __init__(
        self,
        lane: DgLanelet,
        model_params: ModelParameters,
        model_geo: VehicleGeometry,
        speed_behavior: Optional[SpeedBehavior] = None,
        pure_pursuit: Optional[PurePursuit] = None,
        return_extra: bool = False,
    ):
        self.ref_lane = lane
        self.speed_controller: SpeedController = SpeedController.from_vehicle_params(model_param=model_params)
        self.speed_behavior: SpeedBehavior = SpeedBehavior() if speed_behavior is None else speed_behavior
        self.steer_controller: SteerController = SteerController.from_vehicle_params(vehicle_param=model_params)
        self.pure_pursuit: PurePursuit = PurePursuit.from_model_geometry(model_geo)
        self.my_name: Optional[PlayerName] = None
        self.return_extra: bool = return_extra
        self._emergency: bool = False
        self._my_obs: Optional[X] = None
        self.lights_test_seq = DgSampledSequence[LightsCmd](timestamps=[0, 2, 4, 6, 8], values=list(LightsValues))

    def on_episode_init(self, init_sim_obs: InitSimObservations):
        self.my_name = init_sim_obs.my_name
        self.speed_behavior.my_name = init_sim_obs.my_name
        self.pure_pursuit.update_path(self.ref_lane)

    def get_commands(self, sim_obs: SimObservations) -> VehicleCommands:
        self._my_obs = sim_obs.players[self.my_name].state
        my_pose: SE2value = SE2_from_xytheta([self._my_obs.x, self._my_obs.y, self._my_obs.psi])

        # update observations
        self.speed_behavior.update_observations(sim_obs.players)
        self.speed_controller.update_measurement(measurement=self._my_obs.vx)
        self.steer_controller.update_measurement(measurement=self._my_obs.delta)
        lanepose = self.ref_lane.lane_pose_from_SE2_generic(my_pose)
        self.pure_pursuit.update_pose(pose=my_pose, along_path=lanepose.along_lane)

        # compute commands
        t = float(sim_obs.time)
        speed_ref, emergency = self.speed_behavior.get_speed_ref(t)
        if emergency or self._emergency:
            # Once the emergency kicks in the speed ref will always be 0
            self._emergency = True
            speed_ref = 0
            self.emergency_subroutine()
        self.pure_pursuit.update_speed(speed=speed_ref)
        self.speed_controller.update_reference(reference=speed_ref)
        acc = self.speed_controller.get_control(t)
        delta_ref = self.pure_pursuit.get_desired_steering()
        self.steer_controller.update_reference(delta_ref)
        ddelta = self.steer_controller.get_control(t)
        return VehicleCommands(acc=acc, ddelta=ddelta, lights=self.lights_test_seq.at_or_previous(sim_obs.time))

    def emergency_subroutine(self) -> VehicleCommands:
        pass

    def on_get_extra(
        self,
    ) -> Optional[DrawableTrajectoryType]:
        if not self.return_extra:
            return None
        _, gpoint = self.pure_pursuit.find_goal_point()
        pgoal = translation_from_SE2(gpoint)
        l = self.pure_pursuit.param.length
        rear_axle = SE2_apply_T2(self.pure_pursuit.pose, np.array([-l / 2, 0]))
        traj = Trajectory(
            timestamps=[0, 1],
            values=[
                VehicleState(x=rear_axle[0], y=rear_axle[1], psi=0, vx=0, delta=0),
                VehicleState(x=pgoal[0], y=pgoal[1], psi=0, vx=1, delta=0),
            ],
        )
        traj_s = [
            traj,
        ]
        colors = [
            "gold",
        ]
        return list(zip(traj_s, colors))
