import torch
import numpy as np
import gym
import copy

from torch_mpc.models.base import Model
from torch_mpc.util.util import dict_map, dict_to

class CliffordKBM(Model):
    """
    Kinematic bicycle model with steering and velocity dynamics
    Additionally model rear steer for clifford
    x = [x, y, th, v, deltaf, deltar]
    u = [throttle, deltaf_target, deltar_target]
    xdot = [
        v * cos(th + beta)
        v * sin(th + beta)
        v * (tan(deltaf) - tan(deltar) cos(beta) / (Lf + Lr))
        K_throttle * thottle - (k_fric + v * k_drag)
        K_delta * (delta_des - delta)
    ]

    beta = atan2(Lf * tan(deltar) + Lr * tan(deltaf) / Lf + Lr)
    """
    def __init__(self, Lf=1.0, Lr=0.7, throttle_lim=[0., 1.0], steer_lim=[-0.3, 0.3], steer_rate_lim=0.25, k_throttle=1.0, k_drag=0.1, k_fric=0.1, w_Kp=10.0, dt=0.1, device='cpu'):
        self.Lf = Lf
        self.Lr = Lr
        self.dt = dt
        self.device = device

        self.u_lb = np.array([throttle_lim[0], steer_lim[0], steer_lim[0]])
        self.u_ub = np.array([throttle_lim[1], steer_lim[1], steer_lim[1]])
        self.x_lb = -np.ones(6).astype(float) * float('inf')
        self.x_ub = np.ones(6).astype(float) * float('inf')

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
        x, y, theta, v, deltaf, deltar = state.moveaxis(-1, 0)
        throttle, deltaf_des, deltar_des = action.moveaxis(-1, 0)

        tan_df = deltaf.tan()
        tan_dr = deltar.tan()
        beta = torch.arctan2(self.Lf * tan_dr + self.Lr * tan_df, torch.ones_like(tan_df) * (self.Lf + self.Lr))

        xd = v * (theta + beta).cos()
        yd = v * (theta + beta).sin()
        thd = v * (tan_df - tan_dr) * beta.cos() / (self.Lf + self.Lr)

        friction = (self.k_fric + self.k_drag * v) * (v.abs() > 1e-2)
        vd = self.k_throttle * throttle - friction

        deltafd = self.w_Kp * (deltaf_des - deltaf)
        deltafd = deltafd.clip(-self.steer_rate_lim, self.steer_rate_lim)

        deltard = self.w_Kp * (deltar_des - deltar)
        deltard = deltard.clip(-self.steer_rate_lim, self.steer_rate_lim)

        #Enforce steering limits
        below_steer_lim = (deltaf < self.steer_lim[0]).float()
        above_steer_lim = (deltaf > self.steer_lim[1]).float()
        deltafd = deltafd.clip(0., 1e10) * below_steer_lim + deltafd * (1.-below_steer_lim)
        deltafd = deltafd.clip(-1e10, 0) * above_steer_lim + deltafd * (1.-above_steer_lim)

        below_steer_lim = (deltar < self.steer_lim[0]).float()
        above_steer_lim = (deltar > self.steer_lim[1]).float()
        deltard = deltard.clip(0., 1e10) * below_steer_lim + deltard * (1.-below_steer_lim)
        deltard = deltard.clip(-1e10, 0) * above_steer_lim + deltard * (1.-above_steer_lim)

        return torch.stack([xd, yd, thd, vd, deltafd, deltard], dim=-1)

    def predict(self, state, action):
        k1 = self.dynamics(state, action)
        k2 = self.dynamics(state + (self.dt/2)*k1, action)
        next_state = state + self.dt*k2

        return next_state

    def rollout(self, state, actions):
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

        return torch.stack([x, y, yaw, v, torch.zeros_like(x), torch.zeros_like(x)], axis=-1)

    def get_actions(self, batch):
        return batch[...,[0,-1]]

    def get_speed(self, states):
        return states[..., 3]

    def to(self, device):
        self.device = device
        return self

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    kbm = CliffordKBM(
        k_throttle=1.0,
        k_drag=0.1,
        k_fric=0.1
    )

    X0 = torch.zeros(kbm.observation_space().shape[0])
    U = torch.zeros(400, kbm.action_space().shape[0])
    U[:, 0] = 1
    U[:200, 1] = 0.05
    U[200:, 1] = -0.05
    U[:100, 2] = 0.05
    U[100:200, 2] = -0.05
    U[200:300, 2] = 0.05
    U[300:, 2] = -0.05

    X = kbm.rollout(X0, U)

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    for i,label in enumerate(['x', 'y', 'theta', 'v', 'deltaf', 'deltar']):
        axs[0].plot(X[:, i], label=label)

    for i, label in enumerate(['v', 'df', 'dr']):
        axs[1].plot(U[:, i], label=label)

    axs[2].plot(X[:, 0], X[:, 1])
    axs[2].scatter(X[::100, 0], X[::100, 1], c='r', marker='x')

    axs[0].legend()
    axs[1].legend()
    axs[2].legend()

    axs[2].set_aspect('equal')
    plt.show()
