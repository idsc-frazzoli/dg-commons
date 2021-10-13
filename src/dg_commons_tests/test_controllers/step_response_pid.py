import os
from decimal import Decimal
from math import pi

import matplotlib.pyplot as plt

from dg_commons import DgSampledSequence
from dg_commons.controllers.speed import SpeedController, SpeedControllerParam
from dg_commons.controllers.steer import SteerControllerParam, SteerController
from sim.models import kmh2ms
from sim.models.vehicle import VehicleCommands
from sim.models.vehicle_dynamic import VehicleModelDyn, VehicleStateDyn


def sim_step_response(model, sp_controller, st_controller):
    t, sim_step = 0, 0.05
    speed_ref = DgSampledSequence[float](timestamps=[0, 10, 20, 30, 40, 50, 60],
                                         values=[0, kmh2ms(20), kmh2ms(0), kmh2ms(50), kmh2ms(0), kmh2ms(130),
                                                 kmh2ms(0)])

    steer_ref = DgSampledSequence[float](timestamps=[0, 10, 20, 30, 40, 50, 60],
                                         values=[0, pi / 12, pi / 6, -pi / 12, 0, -pi / 12, 0])

    times, speeds, accs, speed_refs, steer, dsteers, steer_refs = [], [], [], [], [], [], []
    while t < 70:
        current_state = model.get_state()

        speeds.append(current_state.vx)
        steer.append(current_state.delta)
        times.append(t)

        sp_controller.update_measurement(measurement=current_state.vx)
        st_controller.update_measurement(measurement=current_state.delta)
        sp_controller.update_reference(reference=speed_ref.at_or_previous(t))
        st_controller.update_reference(reference=steer_ref.at_or_previous(t))

        speed_refs.append(speed_ref.at_or_previous(t))
        steer_refs.append(steer_ref.at_or_previous(t))

        acc = sp_controller.get_control(t)
        dsteer = st_controller.get_control(t)
        accs.append(acc)
        dsteers.append(dsteer)

        cmds = VehicleCommands(acc=acc, ddelta=dsteer)
        model.update(commands=cmds, dt=Decimal("0.1"))
        t += sim_step
        # update observations

    # do plot
    fig, (ax1, ax2,ax3,ax4) = plt.subplots(4)
    fig.suptitle("Step Response PID Controller for Vehicle Speed")
    ax1.plot(times, speeds, label="actual speed")
    ax1.plot(times, speed_refs, "r", label="ref. speed")
    ax2.plot(times, accs)
    ax3.plot(times, steer, label="actual steer")
    ax3.plot(times, steer_refs, "r", label="ref. steer ")
    ax4.plot(times, dsteers)

    ax1.set(ylabel='Velocity')
    ax2.set(xlabel='Time', ylabel='Acceleration Command')
    ax3.set(ylabel='Steering angle')
    ax2.set(xlabel='Time', ylabel='Steering derivative')

    ax1.legend()
    ax3.legend()
    plt.savefig("out/stepresponse")


if __name__ == '__main__':
    try:
        os.mkdir("out/")
    except FileExistsError:
        pass
    # speed pid
    speed_kp: float = 1
    speed_ki: float = 0.01
    speed_kd: float = 0.1

    # steer pid
    steer_kp: float = 1
    steer_ki: float = 0.01
    steer_kd: float = 0.1

    sp_controller_param: SpeedControllerParam = SpeedControllerParam(kP=speed_kp, kI=speed_ki, kD=speed_kd)
    st_controller_param: SteerControllerParam = SteerControllerParam(kP=steer_kp, kI=steer_ki, kD=steer_kd)
    sp_controller = SpeedController(sp_controller_param)
    st_controller = SteerController(st_controller_param)

    "Kinematic Model"
    # x_0 = VehicleState(0, 0, 0, 0, 0)
    # model = VehicleModel.default_car(x_0)
    "Dynamic Model"
    x_0 = VehicleStateDyn(0, 0, 0, 0, 0, 0, 0)
    model = VehicleModelDyn.default_car(x_0)

    sim_step_response(model, sp_controller, st_controller)
