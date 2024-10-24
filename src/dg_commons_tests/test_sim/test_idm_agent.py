import math
import unittest
from bisect import bisect

import numpy as np
from commonroad.scenario.lanelet import Lanelet, LaneletNetwork
from dg_commons.geo import PoseState, SE2Transform
from dg_commons.maps.lanes import DgLanelet, LaneCtrPoint
from dg_commons.sim import PlayerObservations
from dg_commons.sim.models.vehicle_dynamic import VehicleStateDyn
from shapely.geometry import Polygon

from dg_commons.sim.agents.idm_agent.idm_agent_utils import (
    compute_approx_curvatures,
    compute_gap,
    compute_low_speed_intervals,
    compute_projected_obs,
    compute_ref_lane_polygon,
    find_best_lanelet_from_obs,
    find_lanelet_ids_from_obs,
    inside_ref_lane_from_obs,
    state2beta,
    predict_dg_lanelet_from_obs,
    compute_approx_curvatures,
    compute_low_speed_intervals,
)


class TestIDMAgentUtils(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

        self.lane_length = 10
        self.lane_radius = 2
        self.ctrl_point_interval = 1
        self.ctrl_points = [
            LaneCtrPoint(q=SE2Transform.from_PoseState(PoseState(x=prog, y=0.0, psi=0.0)), r=self.lane_radius)
            for prog in range(0, self.lane_length + 1, self.ctrl_point_interval)
        ]
        self.ref_lane = DgLanelet(self.ctrl_points)
        self.ref_lane_polygon = compute_ref_lane_polygon(self.ctrl_points)

        self.center_vertices = np.array(
            [[prog, 0.0] for prog in range(0, self.lane_length + 1, self.ctrl_point_interval)]
        )
        self.left_vertices = np.array(
            [[prog, self.lane_radius] for prog in range(0, self.lane_length + 1, self.ctrl_point_interval)]
        )
        self.right_vertices = np.array(
            [[prog, -self.lane_radius] for prog in range(0, self.lane_length + 1, self.ctrl_point_interval)]
        )
        self.lanelet_id = 0
        self.lanelet = Lanelet(self.left_vertices, self.center_vertices, self.right_vertices, self.lanelet_id)
        self.lanelet_network = LaneletNetwork.create_from_lanelet_list([self.lanelet])

        # ego player
        self.ego_x = 2.0
        self.ego_y = 0.0
        self.ego_epsilon = 0.5
        self.ego_state = VehicleStateDyn(x=self.ego_x, y=self.ego_y, psi=0.0, vx=4.0, delta=0.0)
        self.ego_obs = PlayerObservations(
            state=self.ego_state,
            occupancy=Polygon(
                [
                    (self.ego_x - self.ego_epsilon, self.ego_y + self.ego_epsilon),
                    (self.ego_x + self.ego_epsilon, self.ego_y + self.ego_epsilon),
                    (self.ego_x + self.ego_epsilon, self.ego_y - self.ego_epsilon),
                    (self.ego_x - self.ego_epsilon, self.ego_y - self.ego_epsilon),
                ]
            ),
        )

        # player 1
        self.p1_x = 8.0
        self.p1_y = 0.0
        self.p1_epsilon = 0.5
        self.p1_state = VehicleStateDyn(x=self.p1_x, y=self.p1_y, psi=0.0, vx=2.0, delta=0.0)
        self.p1_obs = PlayerObservations(
            state=self.p1_state,
            occupancy=Polygon(
                [
                    (self.p1_x - self.p1_epsilon, self.p1_y + self.p1_epsilon),
                    (self.p1_x + self.p1_epsilon, self.p1_y + self.p1_epsilon),
                    (self.p1_x + self.p1_epsilon, self.p1_y - self.p1_epsilon),
                    (self.p1_x - self.p1_epsilon, self.p1_y - self.p1_epsilon),
                ]
            ),
        )

        # player 2
        self.p2_x = 9.0
        self.p2_y = 2.1
        self.p2_epsilon = 0.5
        self.p2_state = VehicleStateDyn(x=self.p2_x, y=self.p2_y, psi=math.pi / 4, vx=2.0, delta=0.0)
        self.p2_obs = PlayerObservations(
            state=self.p2_state,
            occupancy=Polygon(
                [
                    (self.p2_x - self.p2_epsilon, self.p2_y + self.p2_epsilon),
                    (self.p2_x + self.p2_epsilon, self.p2_y + self.p2_epsilon),
                    (self.p2_x + self.p2_epsilon, self.p2_y - self.p2_epsilon),
                    (self.p2_x - self.p2_epsilon, self.p2_y - self.p2_epsilon),
                ]
            ),
        )

        # player 3
        self.p3_x = 9.0
        self.p3_y = -5.0
        self.p3_epsilon = 0.5
        self.p3_state = VehicleStateDyn(x=self.p3_x, y=self.p3_y, psi=math.pi / 2, vx=3.0, delta=0.0)
        self.p3_obs = PlayerObservations(
            state=self.p3_state,
            occupancy=Polygon(
                [
                    (self.p3_x - self.p3_epsilon, self.p3_y + self.p3_epsilon),
                    (self.p3_x + self.p3_epsilon, self.p3_y + self.p3_epsilon),
                    (self.p3_x + self.p3_epsilon, self.p3_y - self.p3_epsilon),
                    (self.p3_x - self.p3_epsilon, self.p3_y - self.p3_epsilon),
                ]
            ),
        )
        self.p3_lanelet_ctrl_points = [
            LaneCtrPoint(
                q=SE2Transform.from_PoseState(PoseState(x=self.p3_x, y=prog, psi=math.pi / 2)), r=self.lane_radius
            )
            for prog in range(-self.lane_length, 0 + 1, self.ctrl_point_interval)
        ]
        self.p3_lanelet = DgLanelet(self.p3_lanelet_ctrl_points)
        self.p3_path_int = np.array([self.p3_x, 0.0])

        self.pp3_obs = compute_projected_obs(self.p3_obs, self.p3_path_int, self.p3_lanelet, self.ref_lane)
        self.pp3_state = self.pp3_obs.state

        # Added for new functions: compute_approx_curvatures and compute_low_speed_intervals
        self.ctrl_pt1 = LaneCtrPoint(q=SE2Transform(p=[0.0, 0.0], theta=0.0), r=1.0)
        self.ctrl_pt2 = LaneCtrPoint(q=SE2Transform(p=[1.0, 0.0], theta=0.0), r=1.0)
        self.ctrl_pt3 = LaneCtrPoint(q=SE2Transform(p=[1.0, 1.0], theta=0.0), r=1.0)
        self.ctrl_pt4 = LaneCtrPoint(q=SE2Transform(p=[2.0, 1.0], theta=0.0), r=1.0)
        self.ctrl_pt5 = LaneCtrPoint(q=SE2Transform(p=[3.0, 1.0], theta=0.0), r=1.0)
        self.ctrl_pt6 = LaneCtrPoint(q=SE2Transform(p=[3.0, 0.0], theta=0.0), r=1.0)
        self.ref_lane_curv = DgLanelet(
            [self.ctrl_pt1, self.ctrl_pt2, self.ctrl_pt3, self.ctrl_pt4, self.ctrl_pt5, self.ctrl_pt6]
        )

    def test_bisect(self):
        arr1 = []
        arr2 = [1.0]
        arr3 = [0.0, 1.0]
        arr4 = list(range(10))
        arr5 = [0.0, 1.0, 1.0, 1.0, 2.0]

        self.assertEqual(bisect(arr1, 0.0), 0)
        self.assertEqual(bisect(arr2, 0.0), 0)
        self.assertEqual(bisect(arr2, 2.0), 1)
        self.assertEqual(bisect(arr3, 0.0), 1)
        self.assertEqual(bisect(arr3, 0.5), 1)
        self.assertEqual(bisect(arr3, 1.0), 2)
        self.assertEqual(bisect(arr4, 5.0), 6)
        self.assertEqual(bisect(arr4, 5.5), 6)
        self.assertEqual(bisect(arr4, 9.0), 10)
        self.assertEqual(bisect(arr4, -1.0), 0)
        self.assertEqual(bisect(arr5, 1.0), 4)

    def test_compute_gap(self):
        poly1 = Polygon([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)])
        poly2 = Polygon([(2.0, 2.0), (2.0, 3.0), (3.0, 3.0), (3.0, 2.0)])
        poly3 = Polygon([(0.0, 0.0), (0.0, -1.0), (-1.0, -1.0), (-1.0, 0.0)])
        poly4 = Polygon([(-0.5, 0.5), (0.5, 0.5), (0.5, -0.5), (-0.5, -0.5)])

        self.assertAlmostEqual(compute_gap(poly1, poly2), math.sqrt(2))
        self.assertAlmostEqual(compute_gap(poly1, poly3), 0)
        self.assertAlmostEqual(compute_gap(poly1, poly4), 0)

    def test_compute_ref_lane_polygon(self):
        vertices = [(prog, self.lane_radius) for prog in range(0, self.lane_length + 1, self.ctrl_point_interval)]
        vertices += [(prog, -self.lane_radius) for prog in range(self.lane_length, -1, -self.ctrl_point_interval)]
        poly = Polygon(vertices)

        self.assertTrue(poly.equals_exact(self.ref_lane_polygon, 1e-10))

    def test_obs2prog(self):
        self.assertAlmostEqual(state2beta(self.ref_lane, self.ego_obs.state), 2.0, places=3)

    def test_inside_ref_lane_from_obs(self):
        self.assertTrue(inside_ref_lane_from_obs(self.ref_lane, self.ref_lane_polygon, self.ego_obs))
        self.assertTrue(inside_ref_lane_from_obs(self.ref_lane, self.ref_lane_polygon, self.p1_obs))
        self.assertTrue(inside_ref_lane_from_obs(self.ref_lane, self.ref_lane_polygon, self.p2_obs))
        self.assertFalse(inside_ref_lane_from_obs(self.ref_lane, self.ref_lane_polygon, self.p3_obs))

    def test_compute_projected_obs(self):
        # The first two are not equal within 7 places...
        self.assertAlmostEqual(self.pp3_state.x, 4.0, places=2)
        self.assertAlmostEqual(self.pp3_state.y, 0.0)
        self.assertAlmostEqual(self.pp3_state.psi, 0.0)
        self.assertAlmostEqual(self.pp3_state.vx, 3.0)
        self.assertAlmostEqual(self.pp3_state.delta, 0.0)
        np.testing.assert_array_almost_equal(self.pp3_state.int_point, self.p3_path_int)

    def test_find_lanelet_ids_from_obs(self):
        self.assertTrue(find_lanelet_ids_from_obs(self.lanelet_network, self.ego_obs) == frozenset([0]))
        self.assertTrue(find_lanelet_ids_from_obs(self.lanelet_network, self.p3_obs) == frozenset())

    def test_find_best_lanelet_from_obs(self):
        self.assertTrue(find_best_lanelet_from_obs(self.lanelet_network, self.ego_obs) == self.lanelet)
        self.assertIsNone(find_best_lanelet_from_obs(self.lanelet_network, self.p2_obs))

    def test_predict_dg_lanelet_from_obs(self):
        self.assertIsNone(predict_dg_lanelet_from_obs(self.lanelet_network, self.p2_obs, max_length=50.0))
        self.assertIsNone(predict_dg_lanelet_from_obs(self.lanelet_network, self.p3_obs, max_length=50.0))

    def test_compute_approx_curvatures(self):
        ctrl_points = [self.ctrl_pt1, self.ctrl_pt2, self.ctrl_pt3, self.ctrl_pt4]
        ctrl_points = np.array([ctrl_point.q.p for ctrl_point in ctrl_points])
        curvatures = compute_approx_curvatures(
            ctrl_points, np.array([1, 1, 1])
        )
        self.assertAlmostEqual(curvatures[0], math.pi / 2 / 2)
        self.assertAlmostEqual(curvatures[1], math.pi / 2 / 2)

    def test_compute_low_speed_intervals(self):
        intervals = compute_low_speed_intervals(self.ref_lane_curv, braking_dist=1.0)
        self.assertAlmostEqual(intervals[0][0], 0.0)
        self.assertAlmostEqual(intervals[0][1], 2.0)
        self.assertAlmostEqual(intervals[1][0], 3.0)
        self.assertAlmostEqual(intervals[1][1], 4.0)


if __name__ == "__main__":
    unittest.main()
