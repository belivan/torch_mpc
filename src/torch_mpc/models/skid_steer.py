import torch
import numpy as np
import gym
import copy

from torch_mpc.models.base import Model
from wheeledsim_rl.util.util import dict_map, dict_to

class SkidSteer(Model):
    """
    Kinematic model for a skid-steered vehicle
    X: [x, y, theta, v, w]
    U: [v_target, w_target]
    Dynamics:
    \dot{X} = [
        v cos(theta)
        v sin(theta)
        w
        Kv * (v_target - v)
        Kw * (w_target - w)
    ]
    """
    default_parameters = {
        'log_a_v': torch.tensor(1.0), #linear scaling on velocity (log-scale)
        'b_v': torch.tensor(0.0), #linear offset on velocity
        'log_a_delta': torch.tensor(1.0), #linear scaling on steer (log-scale)
        'b_delta': torch.tensor(0.0), #linear offset on steer
        'log_K_v': torch.tensor(1.0), #p gain on velocity controller (log-scale)
        'log_K_w': torch.tensor(1.0) #p gain on steering controller (log-scale)
    }

    def __init__(self, v_lim=[-3., 3.], w_lim=[-1., 1.], dt=0.1, device='cpu'):
        self.dt = dt
        self.device = device

        self.u_lb = np.array([v_lim[0], w_lim[0]])
        self.u_ub = np.array([v_lim[1], w_lim[1]])
        self.x_lb = -np.ones(5).astype(float) * float('inf')
        self.x_ub = np.ones(5).astype(float) * float('inf')
        self.v_lim = v_lim
        self.w_lim = w_lim

        self.parameters = copy.deepcopy(self.default_parameters)

    def update_parameters(self, parameters):
        assert all([k in self.parameters.keys() for k in parameters]), "Got illegal key. Valid keys are {}".format(list(self.parameters.keys()))

        self.parameters.update(parameters)
        self.parameters = dict_to(self.parameters, self.device)

    def observation_space(self):
        return gym.spaces.Box(low=self.x_lb, high=self.x_ub)

    def action_space(self):
        return gym.spaces.Box(low=self.u_lb, high=self.u_ub)

    def dynamics(self, state, action):
        x, y, theta, v, w = state.moveaxis(-1, 0)
        v_target, w_target = action.moveaxis(-1, 0)

        xd = v * theta.cos()
        yd = v * theta.sin()
        thd = w
        vd = self.parameters['log_K_v'] * (v_target - v)
        wd = self.parameters['log_K_w'] * (w_target - w)

        #Enforce limits
        below_v_lim = (v < self.v_lim[0]).float()
        above_v_lim = (v > self.v_lim[1]).float()
        vd = vd.clip(0., 1e10) * below_v_lim + vd * (1.-below_v_lim)
        vd = vd.clip(-1e10, 0) * above_v_lim + vd * (1.-above_v_lim)

        below_w_lim = (w < self.w_lim[0]).float()
        above_w_lim = (w > self.w_lim[1]).float()
        wd = wd.clip(0., 1e10) * below_w_lim + wd * (1.-below_w_lim)
        wd = wd.clip(-1e10, 0) * above_w_lim + wd * (1.-above_w_lim)

        return torch.stack([xd, yd, thd, vd, wd], dim=-1)

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
        v = state[..., 7:9]
        w = state[..., -1]

        vel = torch.linalg.norm(v, axis=-1)
        yaw = self.quat_to_yaw(q)

        return torch.stack([x, y, yaw, vel, w], axis=-1)

    def get_actions(self, batch):
        return batch[...,[0,-1]]

    def to(self, device):
        self.parameters = dict_to(self.parameters, device)
        self.device = device
        return self

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    kbm = SkidSteer()

    X0 = torch.zeros(kbm.observation_space().shape[0])
    U = torch.zeros(100, kbm.action_space().shape[0])
    U[:, 0] = 1
    U[:50, 1] = 1
    U[50:100, 1] = -1
    U[100:150, 1] = 1
    U[150:, 1] = -1

    X = kbm.rollout(X0, U)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    for i,label in enumerate(['x', 'y', 'theta']):
        axs[0].plot(X[:, i], label=label)

    axs[1].plot(X[:, 0], X[:, 1])
    axs[1].scatter(X[0, 0], X[0, 1], marker='x', c='r')
    axs[0].legend()
    axs[1].legend()
    axs[1].set_aspect('equal')
    plt.show()
