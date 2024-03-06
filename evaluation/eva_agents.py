import cvxpy as cp
import numpy as np
import torch

try:
    from kine_car_nmpc import *
except:
    print("ACADOS not installed")


class Drone_MPC_Agent:
    def __init__(self, control_freq) -> None:
        self.horizon = 20
        self.control_freq = control_freq
        self.dt = 1 / control_freq
        # target state
        self.x_ref = np.array([0, 0.5])
        # model parameters
        self.A = np.array([[1, 0],
                           [0, 1]])
        self.B = np.array([[self.dt, 0],
                           [0, self.dt]])
        # control parameters
        self.P = np.eye(2)
        self.Q = np.eye(2)

        # constraint parameters
        self.D = np.block([[np.eye(2)],
                           [-np.eye(2)]])
        # dynamics contr.
        self.u_lim = np.ones(4) * 0.20
        # obstacle constr.
        self.area_left = np.array([-1.1, 1.8, 4, -0.2])
        self.area_middle = np.array([0, 1.8, 1, -1])
        self.area_right = np.array([1, 1.3, 0.5, -0.2])

    def get_action(self, state):
        x_init = state[:2]

        area_constr = None
        for area in [self.area_left, self.area_middle, self.area_right]:
            if (self.D @ x_init <= area).all():
                area_constr = area
                break

        # create the optimization problem
        x = cp.Variable((2, self.horizon + 1))
        u = cp.Variable((2, self.horizon))
        lamda = cp.Variable(nonneg=True)
        cost = 0
        constr = []
        cost += 10000 * lamda
        for k in range(self.horizon):
            cost += cp.quad_form(x[:, k] - self.x_ref, self.P)
            cost += cp.quad_form(u[:, k], self.Q)
            constr.append(x[:, k+1] == self.A @ x[:, k] + self.B @ u[:, k])
            constr.append(self.D @ u[:, k] <= self.u_lim)
            constr.append(self.D @ x[:, k] <= (area_constr + lamda))
        constr.append(x[:, 0] == x_init)
        # constr.append(cp.norm(u, 2) >= 0.02)
        problem = cp.Problem(cp.Minimize(cost), constr)
        problem.solve()
        # print(x[0, :].value)
        # print(u[0, :].value)
        u = u[:, 0].value
        return u


class Drone_CLF_CBF_Agent:
    def __init__(self, control_freq) -> None:
        self.control_freq = control_freq
        self.dt = 1 / control_freq
        # constraint parameters
        self.D = np.block([[np.eye(2)],
                           [-np.eye(2)]])
        self.u_lim = np.ones(4) * 0.20
        self.x_ref = np.array([0, 0.5])
        self.a = 0.1

    def get_action(self, state):
        x_init = state[:2]
        CLF_constr = np.array(
            [2 * (x_init[0] - self.x_ref[0]), 2 * (x_init[1] - self.x_ref[1])])
        if (x_init[0] < -1):
            CBF_constr = np.array([-self.a / ((self.a + x_init[0] + 1) * (x_init[0] + 1)),
                                   0])
        elif (x_init[0] < -0.5 and x_init[0] > -1 and x_init[1] > 1.3):
            CBF_constr = np.array([-self.a / ((self.a + x_init[0] + 0) * (x_init[0] + 0)),
                                   -self.a / ((self.a + x_init[0] - 1) * (x_init[0] - 1))])
        elif (x_init[0] < -0.5 and x_init[0] > -1):
            CBF_constr = np.array([0,
                                   -self.a / ((self.a + x_init[0] - 1) * (x_init[0] - 1))])
        else:
            CBF_constr = np.array([0,
                                   0])
        # create the optimization problem
        u = cp.Variable((2, 1))
        cost = 0
        constr = []
        constr.append(self.D @ u[:, 0] <= self.u_lim)
        constr.append(CLF_constr @ u[:, 0] <= 0)
        constr.append(CBF_constr @ u[:, 0] <= 0)
        problem = cp.Problem(cp.Minimize(cost), constr)
        problem.solve()
        u = u[:, 0].value
        return u


class Car_NMPC_Agent(object):
    def __init__(self, control_freq) -> None:
        self.dt = 1 / control_freq
        self.horizon = 50
        self.param = MPC_Formulation_Param()
        self.param.set_horizon(dt=self.dt, N=self.horizon)
        self.solver = acados_mpc_solver_generation(
            self.param, collision_avoidance=False)

    def get_action(self, state):
        x_init = state.copy()
        self.solver.set(0, "lbx", x_init)
        self.solver.set(0, "ubx", x_init)
        status = self.solver.solve()
        if status != 0:
            print("acados returned status {} in closed loop iteration {}.".format(
                status))
        u = self.solver.get(0, "u")
        return u


class RL_Agent():
    def __init__(self, PATH) -> None:
        self.policy = torch.load(PATH, map_location='cpu')
        self.device = torch.device("cpu")

    def get_action(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action, _, _ = self.policy.sample(state)
        action = action.detach().cpu().numpy()[0]
        return action


if __name__ == "__main__":
    agent = Car_NMPC_Agent(ctrl_freq=10)
    state = np.array([0, 0, 0])
    action = agent.get_action(state)
    print(action)
