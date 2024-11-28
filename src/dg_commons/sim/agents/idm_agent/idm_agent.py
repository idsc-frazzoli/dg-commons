from dataclasses import dataclass, replace
from math import cos, exp, floor, sqrt
from typing import Any, Mapping, Optional

import numpy as np
from commonroad.scenario.lanelet import Lanelet, LaneletNetwork
from commonroad.scenario.traffic_light import TrafficLight, TrafficLightState
from commonroad.scenario.traffic_sign import SupportedTrafficSignCountry
from commonroad.scenario.traffic_sign_interpreter import TrafficSignInterpreter
from geometry import SE2value, translation_angle_from_SE2
from shapely.geometry import LineString, MultiLineString, MultiPoint, Point, Polygon

from dg_commons import PlayerName
from dg_commons.controllers.pure_pursuit import PurePursuit
from dg_commons.controllers.steer import SteerController
from dg_commons.maps.lanes import DgLanelet
from dg_commons.planning.trajectory import Trajectory
from dg_commons.sim import InitSimObservations, PlayerObservations, SimObservations
from dg_commons.sim.agents.agent import Agent
from dg_commons.sim.goals import RefLaneGoal
from dg_commons.sim.models import extract_pose_from_state
from dg_commons.sim.models.vehicle import VehicleCommands, VehicleState
from dg_commons.sim.models.vehicle_dynamic import VehicleStateDyn
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from dg_commons.sim.models.vehicle_utils import VehicleParameters
from .idm_agent_utils import (
    apply_speed_constraint,
    clip_cmd_value,
    compute_gap,
    compute_low_speed_intervals,
    compute_projected_obs,
    compute_ref_lane_polygon,
    find_lanelet_ids_from_obs,
    inside_ref_lane_from_obs,
    predict_dg_lanelet_from_obs,
    smooth_lanelet,
    state2beta,
    state2beta_duo,
)
from .utils_structures import AV, OnGetExtraAV
from .vehicle_projected import VehicleStatePrj


@dataclass
class IDMParams:
    """This data class is used for storing parameters in the IDM.

    :param v0: desired speed (often set to be the speed limit)
    :param T: safe time headway
    :param s0: linear jam distance
    :param s1: non-linear jam distance (optional)
    :param a: maximum acceleration
    :param b: comfortable deceleration
    :param delta: acceleration exponent

    Default values (except the maximum acceleration) are taken from [this article]
    (https://traffic-simulation.de/info/info_IDM.html).
    The maximum acceleration is taken from
    "Salles et al., Extending the Intelligent Driver Model in SUMO and Verifying the Drive Off Trajectories with Aerial
    Measurements".
    The non-linear jam distance is taken from
    "Malinauskas, The Intelligent Driver Model: Analysis and Application to Adaptive Cruise Control".
    """

    v0: float = 80 / 3.6
    T: float = 1.2
    s0: float = 2.0
    s1: float = 3.0
    a: float = 5.0
    b: float = 3.0
    delta: int = 4

    @classmethod
    def from_aggressiveness(cls, aggr: float, min_percent: float = 0.1) -> "IDMParams":
        """Returns a set of IDM parameters with certain aggressiveness.

        :param min_percent:
        :param aggr: aggressiveness in (0, 1)
        :return: IDM parameters
        """
        return cls(
            v0=cls.v0 * (1 + aggr),
            T=cls.T * (1 - min_percent) * (1 - aggr) + cls.T * min_percent,
            s0=cls.s0 * (1 - min_percent) * (1 - aggr) + cls.s0 * min_percent,
            a=cls.a * (1 + aggr),
            b=cls.b * (1 + aggr),
        )


@dataclass(frozen=True)
class GIDMParams:
    """This data class is used for storing parameters in the GIDM.

    :param sigma: one of the blending-in parameters -- decline factor
    :param TTI_desired: one of the blending-in parameters -- desired time-to-intersection (TTI)
    :param c: the pressure from back parameter -- decaying coefficient (not implemented)

    These values are taken from
    "Kreutz et al., Analysis of the Generalized Intelligent Driver Model (GIDM) for Uncontrolled Intersections"
    The following parameters are not really parameters in GIDM, but they are related.

    :param max_length: max length of the predicted DgLanelets
    :param max_attention_dist: players further (beeline) than max_attention_distance are not considered at all
    :param proj_detecting_length: the length along the lane to detect projected players
    """

    sigma: float = 1.5
    TTI_desired: float = 2.0
    c: float = 0.5
    max_length: float = 100
    max_attention_dist: float = 40
    proj_detecting_length: float = 50

    @classmethod
    def from_aggressiveness(cls, aggr: float) -> "GIDMParams":
        # TODO: is it reasonable to make all the changes linear (params in IDM and GIDM)?
        # TODO: do we need a min_percent here?
        return cls(proj_detecting_length=cls.proj_detecting_length * (1 - aggr))

    @classmethod
    def from_illegal_precedence(cls, new_proj_detecting_length: float = 5.0) -> "GIDMParams":
        # TODO: is the value reasonable?
        return cls(proj_detecting_length=new_proj_detecting_length)


@dataclass(frozen=True)
class SlowTurningParams:
    """Parameters for slowing down before a (sharp) turn.

    :param braking_dist: the braking distance before the turn
    :param max_allowed_curvature: the max allowed curvature such that a braking is not necessary
    :param v0: the desired speed for slow turning
    #fixme maybe make lookahed velocity dependent? and v0 curvature dependent?
    """

    braking_dist: float = 20.0
    max_allowed_curvature: float = 0.01
    v0: float = 20.0 / 3.6

    @classmethod
    def from_aggressiveness(cls, aggr: float) -> "SlowTurningParams":
        """Returns a set of SlowTurningParams with certain aggressiveness.

        :param aggr: aggressiveness in (0, 1)
        :return: SlowTurningParams
        """
        return cls(v0=cls.v0 * (1 + aggr))


class IDMAgent(Agent):
    """This agent class takes its policy from the Intelligent Driver Model."""

    vehicle_params: VehicleParameters
    my_name: PlayerName
    lanelet_network: LaneletNetwork
    _country_id: Any
    ref_lane: DgLanelet
    ref_lane_polygon: Polygon
    my_state: VehicleState
    my_pose: SE2value
    my_prog: float
    model_geo: VehicleGeometry
    pure_pursuit: PurePursuit
    steer_controller: SteerController
    low_speed_intervals: list[list[float]]
    _players: dict[PlayerName, PlayerObservations]
    _sim_obs: SimObservations

    def __init__(
        self,
        aggressiveness: float = 0.0,
        respect_precedence: bool = True,
        respect_speed_limit: bool = True,
        gidm_params: GIDMParams = GIDMParams(),
    ):
        self.aggressiveness = aggressiveness
        self.respect_precedence = respect_precedence
        self.respect_speed_limit = respect_speed_limit

        self.idm_params: IDMParams = IDMParams.from_aggressiveness(self.aggressiveness)
        self.slow_turning_params: SlowTurningParams = SlowTurningParams.from_aggressiveness(self.aggressiveness)
        self.gidm_params: GIDMParams = gidm_params

    def on_episode_init(self, init_sim_obs: InitSimObservations):
        self.my_name = init_sim_obs.my_name
        if isinstance(init_sim_obs.goal, RefLaneGoal):
            self.ref_lane = smooth_lanelet(init_sim_obs.goal.ref_lane)
        else:
            raise ValueError("IDMAgent only supports RefLaneGoal.")
        self.low_speed_intervals = compute_low_speed_intervals(
            self.ref_lane,
            braking_dist=self.slow_turning_params.braking_dist,
            max_allowed_curvature=self.slow_turning_params.max_allowed_curvature,
        )
        self.vehicle_params = replace(init_sim_obs.model_params, vx_limits=(0, 120 / 3.6))
        self.model_geo = init_sim_obs.model_geometry
        self.pure_pursuit = PurePursuit.from_model_geometry(self.model_geo)
        self.pure_pursuit.update_path(self.ref_lane)
        self.steer_controller = SteerController.from_vehicle_params(vehicle_param=self.vehicle_params)
        self.ref_lane_polygon: Polygon = compute_ref_lane_polygon(self.ref_lane.control_points)
        if not self.ref_lane_polygon.is_valid:
            Warning(
                f"The polygon of the reference lane is invalid, whose vertices are: {self.ref_lane_polygon.exterior.xy}"
            )
            self.ref_lane_polygon = self.ref_lane_polygon.buffer(0)

        if init_sim_obs.dg_scenario is not None:
            self.lanelet_network = init_sim_obs.dg_scenario.lanelet_network
            if init_sim_obs.dg_scenario.scenario is not None:
                self._country_id = init_sim_obs.dg_scenario.scenario.scenario_id.country_id

    def _compute_desired_gap(self, v: float, v_diff: float, nonlinear: bool = False) -> float:
        """Computes the desired gap according to the formula of the IDM.

        :param v: tangential velocity of this vehicle
        :param v_diff: tangential velocity difference between the two vehicles
        :param nonlinear: whether to add non-linear jam distance, defaults to False
        :return: the desired gap
        """
        s2 = v * self.idm_params.T + v * v_diff / (2 * sqrt(self.idm_params.a * self.idm_params.b))
        s_desired = self.idm_params.s0 + max((0, s2))

        if nonlinear:
            s_desired += self.idm_params.s1 * sqrt(v / self.idm_params.v0)

        return s_desired

    def _generate_name2beta_dict(self, players: Mapping[PlayerName, PlayerObservations]) -> dict[PlayerName, float]:
        """Generates a dictionary mapping player name to its beta duo along this lanelet.
        The beta duo means (beta, beta of the intersection point or None).
        Only players on this lanelet and being ahead of the current player are recorded.

        :param players: players (including virtual players) in the simulation
        :return: the dictionary
        """

        def _is_relevant_to_me(_my_beta: float, obs: PlayerObservations) -> bool:
            return (
                inside_ref_lane_from_obs(self.ref_lane, self.ref_lane_polygon, obs)
                and state2beta(self.ref_lane, obs.state) > _my_beta
            )

        my_beta = self.ref_lane.beta_from_along_lane(self.my_prog)
        # Filtered players may contain the agent itself because of numerical errors.
        filtered_players = {
            k: state2beta_duo(self.ref_lane, v.state)
            for k, v in players.items()
            if _is_relevant_to_me(my_beta, v) and k != self.my_name
        }

        return filtered_players

    def _find_leading_player(self) -> Optional[PlayerName]:
        """Finds the leading player of the current player.
        When we compare a projected player with a normal player,
        we reset the projected player's position at the intersection point.
        When we compare projected players only, we use their projected positions.

        :return: name of the leading player or None
        """
        names2betas = self._generate_name2beta_dict(self._players)

        if bool(names2betas):
            # If the dictionary is not empty, sort the dictionary.
            names2betas = sorted(names2betas.items(), key=lambda item: item[1][0])
            # When we compare a projected player with a normal player,
            # we reset the projected player's position at the intersection point.
            # When we compare projected players only, we use their projected positions.
            # The sort function in python is stable.
            # So we could sort twice to achieve the requirements above.

            # Sort the dictionary again.
            # The key function produces beta of the intersection point for projected players
            # and beta of the player itself for normal players.
            names2betas = dict(
                sorted(names2betas, key=lambda item: item[1][1] if item[1][1] is not None else item[1][0])
            )

            return next(iter(names2betas))
        else:
            return None

    def _add_virtual_players(self, t: float) -> None:
        """Adds traffic lights, stop signs, etc. to player list as virtual players.
        (TODO: currently only traffic lights are added)

        :param t: simulation time
        :return: new player list
        """

        def _filter_traffic_lights(traffic_light: TrafficLight, time_step: int) -> bool:
            state = traffic_light.get_state_at_time_step(time_step)
            state_condition = (
                state == TrafficLightState.RED
                or state == TrafficLightState.RED_YELLOW
                or state == TrafficLightState.YELLOW
            )

            return traffic_light.active and state_condition

        def _filter_lanelets(lanelet: Lanelet, lanelet_network: LaneletNetwork, t: float) -> bool:
            traffic_lights = (lanelet_network.find_traffic_light_by_id(tl_id) for tl_id in lanelet.traffic_lights)
            time_step: int = floor(t / 0.1)  # assumes time step is 0.1 for commonroad timeseries
            traffic_lights = [tl for tl in traffic_lights if _filter_traffic_lights(tl, time_step)]
            return bool(traffic_lights) and lanelet.stop_line is not None

        lanelets: list[Lanelet] = self.lanelet_network.lanelets
        lanelets = [lanelet for lanelet in lanelets if _filter_lanelets(lanelet, self.lanelet_network, t)]

        # Epsilon is half of the edge length of the virtual polygon.
        epsilon = 0.01

        for lanelet in lanelets:
            # "TL" stands for traffic light.
            name = f"TL{lanelet.lanelet_id}"

            x, y = (lanelet.stop_line.start + lanelet.stop_line.end) / 2
            virtual_state = VehicleStateDyn(x=x, y=y, psi=0.0, vx=0.0, delta=0.0)

            virtual_occupancy = Polygon(
                [
                    (x + epsilon, y + epsilon),
                    (x - epsilon, y + epsilon),
                    (x - epsilon, y - epsilon),
                    (x + epsilon, y - epsilon),
                ]
            )

            virtual_player_obs = PlayerObservations(state=virtual_state, occupancy=virtual_occupancy)
            self._players[PlayerName(name)] = virtual_player_obs

    def _add_projected_players(self) -> None:
        """Adds players on other Lanelets as projected players.
        This policy boosts safety at uncontrolled intersections.
        """

        def _generate_truncated_path(
            _player_obs: PlayerObservations, dg_lanelet: DgLanelet, max_length: float
        ) -> list[Point]:
            state = _player_obs.state
            beta = state2beta(dg_lanelet, _player_obs.state)

            max_beta = dg_lanelet.beta_from_along_lane((dg_lanelet.along_lane_from_beta(beta) + max_length))
            max_beta_in_the_lane = len(dg_lanelet.control_points) - 1

            beta = floor(beta)
            path = [
                Point(*ctrl_point.q.p)
                for i, ctrl_point in enumerate(dg_lanelet.control_points)
                if i > beta and i < max_beta
            ]
            path.insert(0, Point(state.x, state.y))

            if max_beta <= max_beta_in_the_lane:
                max_beta_pos = dg_lanelet.center_point_fast_SE2Transform(max_beta).p
                path.append(Point(max_beta_pos[0], max_beta_pos[1]))

            return path

        def _generate_path(_player_obs: PlayerObservations, dg_lanelet: DgLanelet) -> list[Point]:
            state = _player_obs.state
            beta = state2beta(dg_lanelet, _player_obs.state)
            beta = floor(beta)
            path = [Point(*ctrl_point.q.p) for i, ctrl_point in enumerate(dg_lanelet.control_points) if i > beta]
            path.insert(0, Point(state.x, state.y))

            return path

        def _point_sort_key(_my_pos: tuple[float, float], pos: tuple[float, float]) -> float:
            return (_my_pos[0] - pos[0]) ** 2 + (_my_pos[1] - pos[1]) ** 2

        my_pos = (self.my_state.x, self.my_state.y)
        my_path = _generate_truncated_path(
            self._players[self.my_name], self.ref_lane, self.gidm_params.proj_detecting_length
        )

        if len(my_path) < 2:
            return

        my_path = LineString(my_path)

        projected_players = dict(self._players)

        for player_name, player_obs in self._players.items():
            if str(player_name).startswith("TL") or inside_ref_lane_from_obs(
                self.ref_lane, self.ref_lane_polygon, player_obs
            ):
                continue

            predicted_dg_lanelet = predict_dg_lanelet_from_obs(
                self.lanelet_network, player_obs, self.gidm_params.max_length
            )

            if predicted_dg_lanelet is None:
                continue

            path = _generate_path(player_obs, predicted_dg_lanelet)

            if len(path) < 2:
                continue

            path_int = my_path.intersection(LineString(path))

            if path_int.is_empty:
                continue
            elif isinstance(path_int, Point):
                point_int = np.array([path_int.x, path_int.y])
            elif isinstance(path_int, MultiPoint):
                # Choose the nearest point to the agent.
                path_int_collection = [(pt.x, pt.y) for pt in list(path_int.geoms)]
                path_int_collection.sort(key=lambda pt: _point_sort_key(my_pos, pt))
                point_int = np.array(path_int_collection[0])
            elif isinstance(path_int, LineString):
                # If the intersection of two paths is a LineString, the two paths are identical in some section.
                # So we could only take the end points into consideration.
                # Actually choosing any point on the LineString will produce the same result.
                path_int_collection = list(path_int.coords)
                path_int_collection.sort(key=lambda pt: _point_sort_key(my_pos, pt))
                point_int = np.array(path_int_collection[0])
            elif isinstance(path_int, MultiLineString):
                # Check the scenario "ZAM_Zip-1_65_T-1".
                path_int_collection = []
                for line_string in list(path_int.geoms):
                    path_int_collection.extend(list(line_string.coords))
                path_int_collection.sort(key=lambda pt: _point_sort_key(my_pos, pt))
                point_int = np.array(path_int_collection[0])
            else:
                Warning(f"Type warning for path intersection: {type(path_int)}.")
                continue

            projected_obs = compute_projected_obs(player_obs, point_int, predicted_dg_lanelet, self.ref_lane)

            # "PP" stands for projected player
            projected_players[PlayerName("P" + str(player_name))] = projected_obs

        self._players = projected_players

    def _set_speed_limit(self) -> None:
        """Sets `self.idm_params.v0` to be the max speed of the current Lanelet.
        If no max speed is found, `self.idm_params.v0` remains unchanged.
        """
        lanelet_ids = find_lanelet_ids_from_obs(self.lanelet_network, self._sim_obs.players[self.my_name])

        if not lanelet_ids:
            return

        if self._country_id is None:
            return

        traffic_sign_interpreter = TrafficSignInterpreter(
            SupportedTrafficSignCountry(self._country_id), self.lanelet_network
        )
        speed_limit = traffic_sign_interpreter.speed_limit(lanelet_ids)

        if speed_limit is None:
            return
        else:
            self.idm_params.v0 = min(self.idm_params.v0, speed_limit)

    def get_commands(self, sim_obs: SimObservations) -> VehicleCommands:
        # update self values
        self._sim_obs = sim_obs
        self.my_state = self._sim_obs.players[self.my_name].state
        self.my_pose = extract_pose_from_state(self.my_state)
        self.my_prog = self.ref_lane.along_lane_from_T2value(np.array([self.my_state.x, self.my_state.y]), fast=True)

        # do computations
        if self.respect_speed_limit:
            self._set_speed_limit()

        def is_too_far_from_me(other_xy: tuple[float, float]) -> bool:
            # filter out players that are too far from me
            return (
                np.linalg.norm(np.array([self.my_state.x, self.my_state.y]) - np.array(other_xy))
                > self.gidm_params.max_attention_dist
            )

        self._players = {
            pn: po for pn, po in self._sim_obs.players.items() if not is_too_far_from_me((po.state.x, po.state.y))
        }
        if self.respect_precedence:
            self._add_virtual_players(float(self._sim_obs.time))

        self._add_projected_players()
        leading_player_name = self._find_leading_player()

        self._update_idm_speed_param()
        acc = self._compute_acc(leading_player_name, blending_in=False)
        ddelta = self._compute_ddelta()

        ddelta = clip_cmd_value(ddelta, (-self.vehicle_params.ddelta_max, self.vehicle_params.ddelta_max))
        acc = apply_speed_constraint(self.my_state.vx, acc, self.vehicle_params)
        acc = clip_cmd_value(acc, self.vehicle_params.acc_limits)

        return VehicleCommands(acc=acc, ddelta=ddelta)

    def _update_idm_speed_param(self) -> None:
        """Helps to slow down the vehicle when turning."""

        is_in_intervals = False
        for interval in self.low_speed_intervals:
            if interval[0] <= self.my_prog <= interval[1]:
                is_in_intervals = True
                break

        if is_in_intervals:
            self.idm_params = replace(self.idm_params, v0=self.slow_turning_params.v0)
        else:
            self.idm_params = IDMParams.from_aggressiveness(self.aggressiveness)

    def _compute_acc(self, leading_player_name: PlayerName, blending_in: bool = False) -> float:
        """Computes the acceleration from the formula.

        :return: the acceleration
        """
        my_polygon = self._players[self.my_name].occupancy
        v = self.my_state.vx

        if v < 0:
            # A negative speed would generate a complex number in the expression.
            # Also, if the vehicle is receding, it should reach max acceleration to
            # move forward.
            v = 0

        if leading_player_name is None:
            # If the leading vehicle does not exist, we set the gap to be infinity
            return self.idm_params.a * (1 - (v / self.idm_params.v0) ** self.idm_params.delta)
        else:
            state_lead = self._players[leading_player_name].state
            position_lead = np.array([state_lead.x, state_lead.y])
            polygon_lead = self._players[leading_player_name].occupancy
            _, q_lead = self.ref_lane.find_along_lane_closest_point_fast(position_lead)
            _, dg_lanelet_psi = translation_angle_from_SE2(q_lead)
            psi_lead = state_lead.psi
            v_lead = state_lead.vx * cos(dg_lanelet_psi - psi_lead)

            v_diff = v - v_lead
            # TODO: this is ok but the gap could be difference of progress...
            s = compute_gap(polygon_lead, my_polygon)
            s_desired = self._compute_desired_gap(v, v_diff, nonlinear=False)

            if s < 0.001:
                s = 0.001

            if blending_in and isinstance(state_lead, VehicleStatePrj):
                int_point = np.array(state_lead.int_point)
                int_point_prog = self.ref_lane.along_lane_from_T2value(int_point, fast=True)

                v = max(0.001, v)

                TTI = (int_point_prog - self.my_prog) / v

                if TTI > self.gidm_params.TTI_desired:
                    leading_coef = exp(-(((TTI - self.gidm_params.TTI_desired) / self.gidm_params.sigma) ** 2) / 2)
                else:
                    leading_coef = 1
            else:
                leading_coef = 1

            return self.idm_params.a * (
                1 - (v / self.idm_params.v0) ** self.idm_params.delta - leading_coef * (s_desired / s) ** 2
            )

    def _compute_ddelta(self) -> float:
        self.pure_pursuit.update_pose(self.my_pose, self.my_prog)
        self.pure_pursuit.update_speed(self.my_state.vx)
        self.steer_controller.update_measurement(measurement=self.my_state.delta)

        t = float(self._sim_obs.time)
        delta_ref = self.pure_pursuit.get_desired_steering()
        self.steer_controller.update_reference(delta_ref)
        ddelta = self.steer_controller.get_control(t)

        return ddelta

    def on_get_extra(self) -> Optional[OnGetExtraAV]:
        if self.my_name == AV:
            along_lanes = np.linspace(self.my_prog, self.my_prog + 20, num=10)
            betas = [self.ref_lane.beta_from_along_lane(along_lane) for along_lane in along_lanes]

            _plan = []
            for beta in betas:
                cp = self.ref_lane.center_point_fast_SE2Transform(beta)
                _plan.append(VehicleState(x=cp.p[0], y=cp.p[1], psi=cp.theta, vx=0, delta=0))

            plan = Trajectory(timestamps=range(len(_plan)), values=_plan)
            return OnGetExtraAV(
                my_type=type(self), sim_obs=self._sim_obs, plan=[(plan, "green")], ref_lane=self.ref_lane
            )
        else:
            return OnGetExtraAV(my_type=type(self))
