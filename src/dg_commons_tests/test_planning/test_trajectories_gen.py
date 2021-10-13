import matplotlib.pyplot as plt

from dg_commons.dynamics import BicycleDynamics
from dg_commons.planning import *
from sim.models.vehicle import VehicleState
from sim.models.vehicle_structures import VehicleGeometry
from sim.models.vehicle_utils import VehicleParameters
from sim.simulator_visualisation import plot_trajectories


def _viz(trajectories, name=""):
    # viz
    fig = plt.figure(figsize=(10, 7), dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_aspect("equal")
    traj_lines, traj_points = plot_trajectories(ax=ax, trajectories=list(trajectories))
    # set the limits
    ax.set_xlim([-1, 15])
    ax.set_ylim([-10, 10])
    # ax.autoscale(True, axis="both", tight=True)
    plt.gca().relim(visible_only=True)
    # ax.autoscale_view()
    # plt.draw()
    plt.savefig(f"out/{name}_debug.png")


def test_generate_motion_primitives():
    vp = VehicleParameters.default_car()
    vg = VehicleGeometry.default_car()

    params = MPGParam(dt=Decimal(".2"),
                      n_steps=3,
                      velocity=(0, 50, 3),
                      steering=(-vp.delta_max, vp.delta_max, 3))
    vehicle = BicycleDynamics(vg=vg, vp=vp)
    mpg = MotionPrimitivesGenerator(param=params,
                                    vehicle_dynamics=vehicle.successor_ivp,
                                    vehicle_param=vp)

    traject = mpg.generate()
    _viz(traject, "MotionPrimitivesGenerator")


def test_commands_sampler():
    vp = VehicleParameters.default_car()
    vg = VehicleGeometry.default_car()

    params = CommandsSamplerParam(dt=Decimal(".5"),
                                  n_steps=1,
                                  acc=(-5, 4, 5),
                                  steer_rate=(-vp.ddelta_max, vp.ddelta_max, 5))
    vehicle = BicycleDynamics(vg=vg, vp=vp)
    mpg = CommandsSampler(param=params,
                          vehicle_dynamics=vehicle.successor_ivp,
                          vehicle_param=vp)

    x0 = VehicleState(x=0, y=0, theta=0, vx=5, delta=0)

    traject = mpg.generate(x0)
    _viz(traject, "CommandsSampler")
