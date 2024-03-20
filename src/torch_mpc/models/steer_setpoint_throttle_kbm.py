import torch
import numpy as np
import gym
import copy

from torch_mpc.models.base import Model
from wheeledsim_rl.util.util import dict_map, dict_to

class SteerSetpointThrottleKBM(Model):
    """
    Kinematic bicycle model with steering and velocity dynamics
    x = [x, y, th, v, delta]
    u = [throttle, delta_target]
    xdot = [
        v * cos(th)
        v * sin(th)
        v * tan(delta) / L
        K_throttle * thottle - (k_fric + v * k_drag)
        K_delta * (delta_des - delta)
    ]

    v = a_v * v + v_b
    delta = a_del * delta + b_del

    Hyperparams to optimize:
        a_v
        b_v
        a_del
        b_del
        K_v
        K_del
    leave wheelbase alone.
    """
    def __init__(self, L=3.0, throttle_lim=[0., 1.0], steer_lim=[-0.3, 0.3], steer_rate_lim=0.25, k_throttle=1.0, k_drag=0.1, k_fric=0.1, w_Kp=10.0, dt=0.1, device='cpu'):
        self.L = L
        self.dt = dt
        self.device = device

        self.u_lb = np.array([throttle_lim[0], steer_lim[0]])
        self.u_ub = np.array([throttle_lim[1], steer_lim[1]])
        self.x_lb = -np.ones(5).astype(float) * float('inf')
        self.x_ub = np.ones(5).astype(float) * float('inf')

        self.steer_lim = steer_lim
        self.steer_rate_lim = steer_rate_lim
        self.k_throttle = k_throttle
        self.k_drag = k_drag
        self.k_fric = k_fric
        self.w_Kp = w_Kp

    def observation_space(self):
        return gym.spaces.Box(low=self.x_lb, high=self.x_ub)

    def action_space(self):
        return gym.spaces.Box(low=self.u_lb, high=self.u_ub)

    def dynamics(self, state, action):
        x, y, theta, v, delta = state.moveaxis(-1, 0)
        throttle, delta_des = action.moveaxis(-1, 0)

        v_actual = v
        delta_actual = delta

        xd = v_actual * theta.cos()
        yd = v_actual * theta.sin()
        thd = v * delta_actual.tan() / self.L

        friction = (self.k_fric + self.k_drag * v) * (v.abs() > 1e-2)
        vd = self.k_throttle * throttle - friction

        deltad = self.w_Kp * (delta_des - delta)
        deltad = deltad.clip(-self.steer_rate_lim, self.steer_rate_lim)

        #Enforce steering limits
        below_steer_lim = (delta < self.steer_lim[0]).float()
        above_steer_lim = (delta > self.steer_lim[1]).float()
        deltad = deltad.clip(0., 1e10) * below_steer_lim + deltad * (1.-below_steer_lim)
        deltad = deltad.clip(-1e10, 0) * above_steer_lim + deltad * (1.-above_steer_lim)

        return torch.stack([xd, yd, thd, vd, deltad], dim=-1)

    def predict(self, state, action):
        k1 = self.dynamics(state, action)
        k2 = self.dynamics(state + (self.dt/2)*k1, action)
        next_state = state + self.dt*k2

        return next_state

    def rollout(self, state, actions,extra_states=None):
        """
        Expected shapes:
            state: [B1 x ... x Bn x xd]
            actions: [B1 x ... x Bn x T x ud]
            returns: [B1 x ... x Bn X T x xd]
        """
        X = []
        curr_state = state
        for t in range(actions.shape[-2]):
            action = actions[..., t, :]
            next_state = self.predict(curr_state, action)
            X.append(next_state)
            curr_state = next_state.clone()
        
        return torch.stack(X, dim=-2)

    def quat_to_yaw(self, q):
        #quats are x,y,z,w
        qx, qy, qz, qw = q.moveaxis(-1, 0)
        return torch.atan2(2 * (qw*qz + qx*qy), 1 - 2 * (qy*qy + qz*qz))

    def get_observations(self, batch):
        state = batch['state']

        if len(state.shape) == 1:
            return self.get_observations(dict_map(batch, lambda x:x.unsqueeze(0))).squeeze()

        x = state[..., 0]
        y = state[..., 1]
        q = state[..., 3:7]
        yaw = self.quat_to_yaw(q)
        v = torch.linalg.norm(state[...,7:10], axis=-1)
        delta = batch['steer_angle'][..., 0] * (30./415.) * (-np.pi/180.)

        return torch.stack([x, y, yaw, v, delta], axis=-1)

    def get_actions(self, batch):
        return batch[...,[0,-1]]

    def get_speed(self, states):
        return states[..., 3]

    def to(self, device):
        self.device = device
        return self

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    kbm = SteerSetpointThrottleKBM(
        k_throttle=1.0,
        k_drag=0.1,
        k_fric=0.1
    )

    X0 = torch.zeros(kbm.observation_space().shape[0])
    U = torch.zeros(400, kbm.action_space().shape[0])
    U[:100, 0] = 1
    U[:100, 1] = 1
#    U[100:200, 1] = -1
#    U[200:300, 1] = 1
#    U[300:, 1] = -1

    X = kbm.rollout(X0, U)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    for i,label in enumerate(['x', 'y', 'theta', 'v', 'delta']):
        axs[0].plot(X[:, i], label=label)

    axs[1].plot(X[:, 0], X[:, 1])
    axs[1].scatter(X[200, 0], X[200, 1], marker='x', c='r')
    axs[0].legend()
    axs[1].legend()
    axs[1].set_aspect('equal')
    plt.show()
