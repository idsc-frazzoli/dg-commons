from typing import Optional, List

import numpy as np
from geometry import SE2_from_xytheta, SE2value, translation_from_SE2

from dg_commons import PlayerName, X, DgSampledSequence, SE2_apply_T2, SE2Transform
from dg_commons.controllers.pure_pursuit import PurePursuit
from dg_commons.controllers.speed import SpeedBehavior, SpeedController
from dg_commons.controllers.steer import SteerController
from dg_commons.maps import DgLanelet, LaneCtrPoint
from dg_commons.planning.sampling_algorithms.anytime_rrt_dubins import AnytimeRRTDubins
from dg_commons.planning.sampling_algorithms.node import Path
from dg_commons.planning.trajectory import Trajectory
from dg_commons.sim import SimObservations, DrawableTrajectoryType, SimTime, logger
from dg_commons.sim.agents.agent import Agent
from dg_commons.sim.models.vehicle import VehicleCommands, VehicleState
from dg_commons.sim.models.vehicle_ligths import LightsCmd, LightsValues


class RealTimePlanLFAgent(Agent):
    """This agent is a simple lane follower tracking the centerline of the given lane
    via a pure pursuit controller. The reference in speed is determined by the speed behavior.
    """

    def __init__(
        self,
        planner: AnytimeRRTDubins,
        dt_plan: SimTime,
        dt_expand_tree: SimTime,
        lane: Optional[DgLanelet] = None,
        speed_controller: Optional[SpeedController] = None,
        speed_behavior: Optional[SpeedBehavior] = None,
        pure_pursuit: Optional[PurePursuit] = None,
        steer_controller: Optional[SteerController] = None,
        return_extra: bool = False,
    ):
        self.ref_lane = lane
        self.speed_controller: SpeedController = SpeedController() if speed_controller is None else speed_controller
        self.speed_behavior: SpeedBehavior = SpeedBehavior() if speed_behavior is None else speed_behavior
        self.steer_controller: SteerController = SteerController() if steer_controller is None else steer_controller
        self.pure_pursuit: PurePursuit = PurePursuit() if pure_pursuit is None else pure_pursuit
        self.my_name: Optional[PlayerName] = None
        self.return_extra: bool = return_extra
        self._emergency: bool = False
        self._my_obs: Optional[X] = None
        self.lights_test_seq = DgSampledSequence[LightsCmd](timestamps=[0, 2, 4, 6, 8], values=list(LightsValues))
        self.planner = planner
        self.last_get_plan_ts: SimTime = SimTime("-Infinity")
        self.dt_plan = dt_plan
        self.dt_expand_tree = dt_expand_tree
        self.last_tree_update = SimTime("-Infinity")

    def on_episode_init(self, my_name: PlayerName):
        self.my_name = my_name
        self.speed_behavior.my_name = my_name
        logger.info("Planning first initial path.")
        path = self.planner.planning()
        while path is None:
            self.planner.expand_tree()
            init_pose = SE2Transform(p=[self.planner.state_initial.x, self.planner.state_initial.y],
                                     theta=self.planner.state_initial.theta)
            path = self.planner.replanning(init_pose)
        logger.info("Found initial path.")
        self.ref_lane = self.get_lane_from_path(path)
        self.pure_pursuit.update_path(self.ref_lane)

    def get_commands(self, sim_obs: SimObservations) -> VehicleCommands:
        self._my_obs = sim_obs.players[self.my_name].state
        my_pose: SE2value = SE2_from_xytheta([self._my_obs.x, self._my_obs.y, self._my_obs.theta])

        t = sim_obs.time

        # update planner
        update_tree: bool = (t - self.last_tree_update) >= self.dt_expand_tree
        if update_tree:
            self.planner.sim_observation = sim_obs
            self.last_tree_update = t
            self.planner.expand_tree()
        update_planner: bool = (t - self.last_get_plan_ts) >= self.dt_plan
        if update_planner:
            self.planner.sim_observation = sim_obs
            self.last_get_plan_ts = t
            path = self.planner.replanning(SE2Transform(p=[self._my_obs.x, self._my_obs.y],
                                                                 theta=self._my_obs.theta))
            if path is None:
                logger.info(f"Emergency stop at {str(t)}.")
                self.speed_behavior.update_observations(sim_obs.players)
                self.speed_controller.update_measurement(measurement=self._my_obs.vx)
                self.steer_controller.update_measurement(measurement=self._my_obs.delta)
                lanepose = self.ref_lane.lane_pose_from_SE2_generic(my_pose)
                self.pure_pursuit.update_pose(pose=my_pose, along_path=lanepose.along_lane)
                t = float(t)
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
            else:
                self._emergency = False
                self.ref_lane = self.get_lane_from_path(path)
                self.pure_pursuit.update_path(self.ref_lane)

        # update observations
        self.speed_behavior.update_observations(sim_obs.players)
        self.speed_controller.update_measurement(measurement=self._my_obs.vx)
        self.steer_controller.update_measurement(measurement=self._my_obs.delta)
        lanepose = self.ref_lane.lane_pose_from_SE2_generic(my_pose)
        self.pure_pursuit.update_pose(pose=my_pose, along_path=lanepose.along_lane)


        # compute commands
        t = float(t)
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
        all_queries_list = self.get_all_queries([node.path for node in self.planner.tree.tree.values()])
        colors_queries = ["blue" for traj in all_queries_list]
        queries = list(zip(all_queries_list, colors_queries))
        trajectories = self.get_best_trajectory(self.planner.path)
        if trajectories is not None:
            trajectories = [trajectories]
            colors_opt = ["gold" for traj in trajectories]
            traj_opt = list(zip(trajectories, colors_opt))
            trajectory = queries + traj_opt
        else:
            trajectory = queries
        _, gpoint = self.pure_pursuit.find_goal_point()
        pgoal = translation_from_SE2(gpoint)
        l = self.pure_pursuit.param.length
        rear_axle = SE2_apply_T2(self.pure_pursuit.pose, np.array([-l / 2, 0]))
        traj = Trajectory(
            timestamps=[0, 1],
            values=[
                VehicleState(x=rear_axle[0], y=rear_axle[1], theta=0, vx=0, delta=0),
                VehicleState(x=pgoal[0], y=pgoal[1], theta=0, vx=1, delta=0),
            ],
        )
        traj_s = [
            traj,
        ]
        colors_s = [
            "green",
        ]
        trajectory = trajectory + list(zip(traj_s, colors_s))
        return trajectory

    def get_lane_from_path(self, path: List[SE2Transform]) -> DgLanelet:
        ctr_points = []
        for p in path:
            ctr_points.append(LaneCtrPoint(p, r=0.01))
        return DgLanelet(ctr_points)

    def get_all_queries(self, path_list: List[List[SE2Transform]]):
        all_queries_list = []
        for p in path_list:
            vs_list = []
            ts_list = []
            ts = 0.0
            for s in p:
                vs = VehicleState(x=s.p[0], y=s.p[1], theta=s.theta, vx=0, delta=0)
                ts_list.append(ts)
                vs_list.append(vs)
                ts += 1
            tra = Trajectory(timestamps=ts_list, values=vs_list)
            all_queries_list.append(tra)

        return all_queries_list

    def get_best_trajectory(self, path: Path):
        if path is None:
            return None
        timestamp = 0.0
        v_state_list = []
        timestamp_list = []
        for p in path.path:
            v_state = VehicleState(x=p.p[0], y=p.p[1], theta=p.theta, vx=0, delta=0)
            timestamp_list.append(timestamp)
            v_state_list.append(v_state)
            timestamp += 1

        return Trajectory(timestamps=timestamp_list, values=v_state_list)

