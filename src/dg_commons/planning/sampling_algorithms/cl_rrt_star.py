from typing import List

import numpy as np
from commonroad.planning.planning_problem import PlanningProblem

from dg_commons import SE2Transform
from dg_commons.planning import PlanningGoal
from dg_commons.planning.sampling_algorithms import pure_pursuit, reeds_shepp_path, unicycle_model
from dg_commons.planning.sampling_algorithms.rrt_star_reeds_shepp import RRTStarReedsShepp
from dg_commons.sim.models.vehicle import VehicleState
from dg_commons.sim.models.vehicle_dynamic import VehicleStateDyn, VehicleModelDyn
from dg_commons.sim.scenarios import DgScenario


class ClosedLoopRRTStar(RRTStarReedsShepp):
    def __init__(self, scenario: DgScenario, planningProblem: PlanningProblem,
                 initial_vehicle_state: VehicleState, goal: PlanningGoal, goal_state: VehicleState,
                 max_iter: int, goal_sample_rate: int, expand_dis: float, path_resolution: float,
                 curvature: float, goal_yaw_th: float, goal_xy_th: float, connect_circle_dist: float,
                 search_until_max_iter: bool, seed: int, target_speed: float, yaw_th: float, xy_th: float,
                 invalid_travel_ratio: float):
        super().__init__(scenario=scenario, planningProblem=planningProblem,
                         initial_vehicle_state=initial_vehicle_state, goal=goal, goal_state=goal_state,
                         max_iter=max_iter, goal_sample_rate=goal_sample_rate,
                         expand_dis=expand_dis, path_resolution=path_resolution,
                         curvature=curvature, goal_yaw_th=goal_yaw_th, goal_xy_th=goal_xy_th,
                         connect_circle_dist=connect_circle_dist,
                         search_until_max_iter=search_until_max_iter, seed=seed)
        self.target_speed = target_speed
        self.yaw_th = yaw_th
        self.xy_th = xy_th
        self.invalid_travel_ratio = invalid_travel_ratio

    def planning(self):
        """
        do planning
        animation: flag for animation on or off
        """
        # planning with RRTStarReedsShepp
        super().planning()

        # generate coruse
        path_indexs = self.get_goal_indexes()

        flag, x, y, yaw, v, t, a, d = self.search_best_feasible_path(
            path_indexs)

        path = []
        for xi, xv in enumerate(x):
            path.append(SE2Transform(p=np.array([x[xi], y[xi]]), theta=yaw[xi]))

        return path

    def search_best_feasible_path(self, path_indexs):

        print("Start search feasible path")

        best_time = float("inf")

        fx, fy, fyaw, fv, ft, fa, fd = None, None, None, None, None, None, None

        # pure pursuit tracking
        for ind in path_indexs:
            path = self.generate_final_course(ind)

            flag, x, y, yaw, v, t, a, d = self.check_tracking_path_is_feasible(
                path)

            if flag and best_time >= t[-1]:
                print("feasible path is found")
                best_time = t[-1]
                fx, fy, fyaw, fv, ft, fa, fd = x, y, yaw, v, t, a, d

        print("best time is")
        print(best_time)

        if fx:
            fx.append(self.goal_node.pose.p[0])
            fy.append(self.goal_node.pose.p[1])
            fyaw.append(self.goal_node.pose.theta)
            return True, fx, fy, fyaw, fv, ft, fa, fd

        return False, None, None, None, None, None, None, None

    def check_tracking_path_is_feasible(self, path):
        cx = np.array([state.p[0] for state in path])
        cy = np.array([state.p[1] for state in path])
        cyaw = np.array([state.theta for state in path])

        goal = [cx[-1], cy[-1], cyaw[-1]]

        cx, cy, cyaw = pure_pursuit.extend_path(cx, cy, cyaw)

        speed_profile = pure_pursuit.calc_speed_profile(
            cx, cy, cyaw, self.target_speed)

        t, x, y, yaw, v, a, d, find_goal = pure_pursuit.closed_loop_prediction(
            cx, cy, cyaw, speed_profile, goal)
        yaw = [reeds_shepp_path.pi_2_pi(iyaw) for iyaw in yaw]

        if not find_goal:
            print("cannot reach goal")

        if abs(yaw[-1] - goal[2]) >= self.yaw_th * 10.0:
            print("final angle is bad")
            find_goal = False

        travel = unicycle_model.dt * sum(np.abs(v))
        origin_travel = sum(np.hypot(np.diff(cx), np.diff(cy)))

        if (travel / origin_travel) >= self.invalid_travel_ratio:
            print("path is too long")
            find_goal = False
        pose_list = []
        for (ix, iy, iyaw) in zip(x, y, yaw):
            pose_list.append(SE2Transform(p=np.array([ix, iy]), theta=iyaw))

        if self.check_collision_with_pose(pose_list):
            print("This path is collision")
            find_goal = False

        return find_goal, x, y, yaw, v, t, a, d

    def get_goal_indexes(self):
        goalinds = []
        for (i, node) in enumerate(self.node_list):
            if self.calc_dist_to_goal(node) <= self.xy_th:
                goalinds.append(i)
        print("OK XY TH num is")
        print(len(goalinds))

        # angle check
        fgoalinds = []
        for i in goalinds:
            if abs(self.node_list[i].pose.theta - self.goal_node.pose.theta) <= self.yaw_th:
                fgoalinds.append(i)
        print("OK YAW TH num is")
        print(len(fgoalinds))

        return fgoalinds

    def check_collision_with_pose(self, pose_list: List[SE2Transform]) -> bool:
        """Check collisions of the planned trajectory with the environment
        :param trajectory: The planned trajectory
        :return: True if at least one collision happened, False otherwise"""
        env_obstacles = self.sim_context.dg_scenario.strtree_obstacles
        collision = False
        for pose in pose_list:
            x0_p1 = VehicleStateDyn(x=pose.p[0], y=pose.p[1], theta=pose.theta,
                                    vx=0.0, delta=0.0)
            p_model = VehicleModelDyn.default_car(x0_p1)
            footprint = p_model.get_footprint()
            assert footprint.is_valid
            p_shape = footprint
            items = env_obstacles.query_items(p_shape)
            for idx in items:
                candidate = self.sim_context.dg_scenario.static_obstacles[idx]
                if p_shape.intersects(candidate.shape):
                    collision = True
        return collision

