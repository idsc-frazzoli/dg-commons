from typing import Tuple, Union, Type

import numpy as np
from dg_commons.sim.models.pedestrian import PedestrianState

from dg_commons.sim.models.vehicle_dynamic import VehicleStateDyn

from dg_commons.sim.models.vehicle import VehicleState

from dg_commons.sim import SimModel, SimTime
from scipy.optimize import minimize

from dg_commons import X, U

__all__ = ["reconstruct_input_transition"]


def position_orientation_objective(
    u: np.array,
    x0: np.array,
    x1: np.array,
    dt: SimTime,
    model_dynamics: SimModel,
    ftol: float = 1e-8,
    e: np.array = np.array([2e-2, 2e-2, 3e-2]),
) -> float:
    """
    Position-Orientation objective function to be minimized for the state transition.

    Simulates the next state using the inputs and calculates the norm of the difference between the
    simulated next state and actual next state. Position, velocity and orientation state fields will
    be used for calculation of the norm.

    :param u: input values
    :param x0: initial state values
    :param x1: next state values
    :param dt: delta time
    :param model_dynamics: the vehicle dynamics model to be used for forward simulation
    :param ftol: ftol parameter used by the optimizer
    :param e: error margin, function will return norm of the error vector multiplied with 100 as cost
        if the input violates the friction circle constraint or input bounds.
    :return: cost
    """
    try:
        x0_adjusted = _adjust_state_bounds(x0, model_dynamics, ftol)
        x1_adjusted = _adjust_state_bounds(x1, model_dynamics, ftol)
        x1_sim = model_dynamics.update(x0_adjusted, u, dt)

        # if the input violates the constraints
        if x1_sim is None:
            return np.linalg.norm(e * 100)

        if isinstance(model_dynamics, PointMassDynamics):
            diff = np.subtract(x1_adjusted, x1_sim)
            cost = np.linalg.norm(diff)
            return cost

        pos_diff = np.subtract(x1_adjusted[:2], x1_sim[:2])
        # steering_diff = _angle_diff(x1_adjusted[2], x1_sim[2])
        vel_diff = x1_adjusted[3] - x1_sim[3]
        orient_diff = _angle_diff(x1_adjusted[4], x1_sim[4])
        diff = np.append(pos_diff, np.array([vel_diff, orient_diff]))
        cost = np.linalg.norm(diff)
        if not isinstance(model_dynamics, PointMassDynamics):
            diff_tmp = diff[:3]
            if np.any(np.greater(np.abs(diff_tmp), e)):
                cost += np.linalg.norm(
                    np.abs(diff_tmp[np.greater(np.abs(diff_tmp), e)]) - e[np.greater(np.abs(diff_tmp), e)]
                )
        return cost

    except VehicleDynamicsException as ex:
        msg = (
            f"An exception occurred during the calculation of position-orientation objective!\n"
            f"x0: {x0}\nx1: {x1}\nu: {u}\nVehicle: {type(model_dynamics)}\ndt: {dt}, ftol: {ftol}"
        )
        raise FeasibilityObjectiveException(msg) from ex


def reconstruct_input_transition(
    x0: Union[VehicleState, VehicleStateDyn, PedestrianState],
    x1: Union[VehicleState, VehicleStateDyn, PedestrianState],
    type_U: Type[U],
    vehicle_dynamics: SimModel,
    dt: float,
    objective=position_orientation_objective,
    criteria=position_orientation_feasibility_criteria,
    ftol: float = 1e-8,
    e: np.array = np.array([2e-2, 2e-2, 3e-2]),
    d: int = 4,
    maxiter: int = 100,
    disp: bool = False,
) -> Tuple[bool, U]:
    """
    Tries to find a valid input for the state transition by minimizing the objective function, and then
    checks if the state simulated by using the reconstructed input is feasible.

    By default, the trajectory feasibility checker will use position-orientation objective function as the
    objective and position-orientation feasibility criteria function will be used for feasibility criteria.

    Objectives can be changed by passing a function with the signature `fun(u: np.array, x0: np.array,
    x1: np.array, dt: float, vehicle_dynamics: VehicleDynamics, ftol: float = 1e-8, e: np.array -> float`

    Feasibility criteria can be changed by passing a function with the signature `fun(x: np.array,
    x_sim: np.array, vehicle_dynamics: VehicleDynamics, e: np.array = np.array([2e-2, 2e-2, 3e-2]),
    d: int = 4) -> bool`

    :param x0: initial state
    :param x1: next state
    :param vehicle_dynamics: the vehicle dynamics model to be used for forward simulation
    :param dt: delta time
    :param objective: callable `fun(u, x0, x1, dt, vehicle_dynamics) -> float`, objective function to be
        minimized in order to find a valid input for state transition
    :param criteria: callable `fun(x1, x_sim, vehicle_dynamics) -> bool`, feasibility criteria to be checked
        between the real next state and the simulated next state
    :param ftol: ftol passed to the minimizer function
    :param e: error margin passed to the feasibility criteria function
    :param d: decimal points where the difference values are rounded up to in order to avoid floating point
        errors set it based on the error margin, i.e e=0.02, d=4
    :param maxiter: maxiter passed to the minimizer function
    :param disp: disp passed to the minimizer function
    :return: True if an input satisfying the tolerance was found, and the constructed input
    """
    try:
        x0_np = x0.as_ndarray()
        x1_np = x1.as_ndarray()
        u0 = np.array([0, 0])

        # Minimize difference between simulated state and next state by varying input u
        u0 = minimize(
            objective,
            u0,
            args=(x0_np, x1_np, dt, vehicle_dynamics, ftol, e),
            options={"disp": disp, "maxiter": maxiter, "ftol": ftol},
            method="SLSQP",
            bounds=vehicle_dynamics.input_bounds,
        ).x

        # Get simulated state using the found inputs
        x1_sim = vehicle_dynamics.forward_simulation(x0_np, u0, dt, throw=False)
        if x1_sim is None:
            msg = (
                f"Minimizer was not able to reconstruct a valid input for the given states!\n"
                f"x0: {x0}\nx1: {x1}\nVehicle: {type(vehicle_dynamics)}\nReconstructed input:{u0}\n"
                f"dt: {dt}, ftol: {ftol}, e: {e}, d: {d}, maxiter: {maxiter}, disp: {disp}"
            )
            if disp:
                print(msg)
            return False, vehicle_dynamics.array_to_input(u0, x0_ts)

        # Check the criteria for the feasibility
        feasible = criteria(x1_np, x1_sim, vehicle_dynamics, e, d)

        return feasible, vehicle_dynamics.array_to_input(u0, x0_ts)

    except (FeasibilityObjectiveException, FeasibilityCriteriaException) as ex:
        msg = (
            f"An exception occurred within the objective of feasibility criteria functions!\n"
            f"x0: {x0}\nx1: {x1}\nVehicle: {type(vehicle_dynamics)}\n"
            f"dt: {dt}, ftol: {ftol}, e: {e}, d: {d}, maxiter: {maxiter}, disp: {disp}"
        )
        raise StateTransitionException(msg) from ex

    except Exception as ex:  # catch any other exception (in order to debug if there is an unexpected error)
        msg = (
            f"An exception occurred during state transition feasibility checking!\n"
            f"x0: {x0}\nx1: {x1}\nVehicle: {type(vehicle_dynamics)}\ndt: {dt}, ftol: {ftol}, "
            f"e: {e}, d: {d}, maxiter: {maxiter}, disp: {disp}"
        )
        raise Exception(msg) from ex


def trajectory_feasibility(
    trajectory: Trajectory,
    vehicle_dynamics: VehicleDynamics,
    dt: float,
    objective=position_orientation_objective,
    criteria=position_orientation_feasibility_criteria,
    ftol: float = 1e-8,
    e: np.array = np.array([2e-2, 2e-2, 3e-2]),
    d: int = 4,
    maxiter: int = 100,
    disp: bool = False,
) -> Tuple[bool, Trajectory]:
    """
    Checks if the given trajectory is feasible for the vehicle model by checking if the state transition is
    feasible between each consecutive state of the trajectory.

    The state_transition_feasibility function will be applied to consecutive states of a given trajectory,
    and the reconstructed inputs will be returned as Trajectory object. If the trajectory was not feasible,
    reconstructed inputs up to infeasible state will be returned.

    ATTENTION: Reconstructed inputs are just approximated inputs for the forward simulation between
    consecutive states n and n+1. Simulating full trajectory from the initial state by using the
    reconstructed inputs can result in a different (but similar) trajectory compared to the real one.
    The reason for this is the small differences between the approximate inputs and the real inputs adding
    up as we simulate further from the initial state.

    By default, the trajectory feasibility checker will use position-orientation objective function as the
    objective and position-orientation feasibility criteria function will be used for feasibility criteria.

    Objectives can be changed by passing a function with the signature `fun(u: np.array, x0: np.array,
    x1: np.array, dt: float, vehicle_dynamics: VehicleDynamics, ftol: float = 1e-8, e: np.array) -> float`

    Feasibility criteria can be changed by passing a function with the signature `fun(x: np.array,
    x_sim: np.array, vehicle_dynamics: VehicleDynamics, e: np.array = np.array([2e-2, 2e-2, 3e-2]),
    d: int = 4) -> bool`

    :param trajectory: trajectory
    :param vehicle_dynamics: the vehicle dynamics model to be used for forward simulation
    :param dt: delta time
    :param objective: callable `fun(u, x0, x1, dt, vehicle_dynamics) -> float`, objective function to be
        minimized in order to find a valid input for state transition
    :param criteria: callable `fun(x1, x_sim, vehicle_dynamics) -> bool`, feasibility criteria to be
        checked between the real next state and the simulated next state
    :param ftol: ftol passed to the minimizer function
    :param e: error margin passed to the feasibility criteria function
    :param d: decimal points where the difference values are rounded up to in order to avoid floating
        point errors set it based on the error margin, i.e e=0.02, d=4
    :param maxiter: maxiter passed to the minimizer function
    :param disp: disp passed to the minimizer function
    :return: True if feasible, and list of constructed inputs as Trajectory object
    """
    trajectory_type = TrajectoryType.get_trajectory_type(trajectory, vehicle_dynamics.vehicle_model)
    if trajectory_type in [TrajectoryType.Input, TrajectoryType.PMInput]:
        raise FeasibilityException("Invalid trajectory type!")

    try:
        reconstructed_inputs = []
        for x0, x1 in zip(trajectory.state_list[:-1], trajectory.state_list[1:]):
            feasible, reconstructed_input = state_transition_feasibility(
                x0, x1, vehicle_dynamics, dt, objective, criteria, ftol, e, d, maxiter, disp
            )
            reconstructed_inputs.append(reconstructed_input)
            if not feasible:
                input_vector = Trajectory(
                    initial_time_step=reconstructed_inputs[0].time_step, state_list=reconstructed_inputs
                )
                return False, input_vector

        input_vector = Trajectory(initial_time_step=reconstructed_inputs[0].time_step, state_list=reconstructed_inputs)
        return True, input_vector

    except StateTransitionException as ex:
        msg = (
            f"An error occurred during feasibility checking!\n"
            f"Vehicle: {type(vehicle_dynamics)}\ndt: {dt}, ftol: {ftol}, "
            f"e: {e}, d: {d}, maxiter: {maxiter}, disp: {disp}"
        )
        raise TrajectoryFeasibilityException(msg) from ex

    except Exception as ex:  # catch any other exception (in order to debug if there is an unexpected error)
        msg = (
            f"An exception occurred during trajectory feasibility checking!\n"
            f"Vehicle: {type(vehicle_dynamics)}\ndt: {dt}, ftol: {ftol}, "
            f"e: {e}, d: {d}, maxiter: {maxiter}, disp: {disp}"
        )
        raise Exception(msg) from ex
