from time import process_time

from numpy import deg2rad

from dg_commons.planning.trajectory import Trajectory
from sim.models.vehicle import VehicleState


def test_trajectory():
    x0_p1 = VehicleState(x=2, y=16, theta=0, vx=5, delta=0)
    x0_p2 = VehicleState(x=22, y=6, theta=deg2rad(90), vx=6, delta=0)
    x0_p3 = VehicleState(x=45, y=22, theta=deg2rad(180), vx=4, delta=0)
    ts = [1, 2, 3]
    t0 = process_time()
    t = Trajectory(ts, [x0_p1, x0_p2, x0_p3])
    t1 = process_time()
    print(t1 - t0)
    assert t.XT == VehicleState
