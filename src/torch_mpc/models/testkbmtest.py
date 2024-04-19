import torch
import numpy as np
import gym
import abc

# from torch_mpc.models.base import Model
# from wheeledsim_rl.util.util import dict_map

class Model(abc.ABC):
    """
    Base model class
    """
    @abc.abstractmethod
    def observation_space(self):
        pass

    @abc.abstractmethod
    def action_space(self):
        pass

    @abc.abstractmethod
    def predict(self, state, action):
        pass

    @abc.abstractmethod
    def rollout(self, state, actions):
        pass

    # @abc.abstractmethod
    # def get_observations(self, batch):
    #     """
    #     Convert a batch of torch data (i.e. 13D odom tensors)
    #     into batches of model states
    #     """
    #     pass

    @abc.abstractmethod
    def get_actions(self, batch):
        """
        Convert a batch of torch data into batches of actions
        """
        pass

class KBM(Model):
    """
    Kinematic bicycle model
    x = [x, y, th] 
    u = [v, delta]
    xdot = [
        v * cos(th)
        v * sin(th)
        L / tan(delta)
    ]
    """
    def __init__(self, L=3.0, min_throttle=0., max_throttle=1., max_steer=0.3, dt=0.1, device='cpu'):
        self.L = L
        self.dt = dt
        self.device = device
        self.u_ub = np.array([max_throttle, max_steer])
        self.u_lb = np.array([min_throttle, -max_steer])

    def observation_space(self):
        low = -np.ones(3).astype(float) * float('inf')
        high = -low
        return gym.spaces.Box(low=low, high=high)

    def action_space(self):
        return gym.spaces.Box(low=self.u_lb, high=self.u_ub)

    def dynamics(self, state, action):
        x, y, th = state.moveaxis(-1, 0)
        v, d = action.moveaxis(-1, 0)
        xd = v * th.cos()
        yd = v * th.sin()
        thd = v * d.tan() / self.L
        return torch.stack([xd, yd, thd], dim=-1)

    def predict(self, state, action):
        k1 = self.dynamics(state, action)
        k2 = self.dynamics(state + (self.dt/2)*k1, action)
        k3 = self.dynamics(state + (self.dt/2)*k2, action)
        k4 = self.dynamics(state + self.dt*k3, action)

        next_state = state + (self.dt/6) * (k1 + 2*k2 + 2*k3 + k4)

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

    # def get_observations(self, batch):
    #     state = batch['state']
    #     if len(state.shape) == 1:
    #         return self.get_observations(dict_map(batch, lambda x:x.unsqueeze(0))).squeeze()

        x = state[..., 0]
        y = state[..., 1]
        q = state[..., 3:7]
        yaw = self.quat_to_yaw(q)
        return torch.stack([x, y, yaw], axis=-1)

    def get_actions(self, batch):
        return batch

    def get_speed(self, states):
        raise NotImplementedError

    def to(self, device):
        self.device = device
        return self
    
if __name__ == '__main__':
    # Sam TODO: write up a test script w/ varying pitches
    # import yaml
    import matplotlib.pyplot as plt
    # from torch_mpc.setup_mpc import setup_mpc

    # config_fp = '/home/atv/physics_atv_ws/src/control/torch_mpc/configs/test_throttle_config.yaml'
    # config = yaml.safe_load(open(config_fp, 'r'))
    # mpc = setup_mpc(config)
    model = KBM()

    x0 = torch.zeros(3)
    U = torch.zeros(50, 2); U[:, 0]=1; U[:, 1]=0.15

    X1 = model.rollout(x0, U)

    x0[-1] = -0.2
    X2 = model.rollout(x0, U)

    x0[-1] = 0.2
    X3 = model.rollout(x0, U)

    plt.plot(X1[:, 0], X1[:, 1], label='flat')
    plt.plot(X2[:, 0], X2[:, 1], label='down')
    plt.plot(X3[:, 0], X3[:, 1], label='up  ')
    plt.gca().legend()
    plt.gca().set_aspect(1.)
    plt.show()