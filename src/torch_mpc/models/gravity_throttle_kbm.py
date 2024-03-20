import torch
import numpy as np
import gym
import rospy
import time
import math
import copy

from torch_mpc.models.base import Model
from torch_mpc.models.colors import plt_colors
from torch_mpc.util.utils import *
g = 9.81
class GravityThrottleKBM(Model): 
    def __init__(self, actuator_model, L=3.0, throttle_lim=[0., 1.0], steer_lim=[-0.52, 0.52], steer_rate_lim=0.45, dt=0.1, device='cpu',  requires_grad=False, state_key="state", pitch_key="pitch", steer_key="steer_angle"):
        """
        Args:
            actuator model: a 6-param model of how throttle and steer commands map to actual throttle/steer (TODO Parv describe better)
        """
        self.L = L
        self.dt = dt
        self.state_key = state_key
        self.pitch_key = pitch_key
        self.steer_key = steer_key
        self.device = device

        self.u_lb = np.array([throttle_lim[0], steer_lim[0]])
        self.u_ub = np.array([throttle_lim[1], steer_lim[1]])
        self.x_lb = -np.ones(5).astype(float) * float('inf')
        self.x_ub = np.ones(5).astype(float) * float('inf')

        self.steer_lim = steer_lim
        self.steer_rate_lim = steer_rate_lim
        self.actuator_model = [math.exp(x) for x in actuator_model]

        self.requires_grad = requires_grad
        self.num_params = len(self.actuator_model)
    
    def observation_space(self):
        return gym.spaces.Box(low=self.x_lb, high=self.x_ub)

    def action_space(self):
        return gym.spaces.Box(low=self.u_lb, high=self.u_ub)

    def update_parameters(self, parameters):
        assert all([k in self.parameters.keys() for k in parameters]), "Got illegal key. Valid keys are {}".format(list(self.parameters.keys()))

        self.parameters.update(parameters)
        self.parameters = dict_to(self.parameters, self.device)
    
    def reset_parameters(self,requires_grad= None):
        if requires_grad == None:
            requires_grad = self.requires_grad
        self.parameters = {k : torch.tensor(self.config['params'][k],requires_grad=requires_grad,device = self.device) for k in self.config['params'].keys()}
        # self.parameters = torch.nn.ParameterDict(self.parameters)

    def dynamics(self, state, action):
#        if self.debug:
#            import pdb;pdb.set_trace()
        x, y, theta, v, delta, pitch = state.moveaxis(-1, 0)
        throttle_og, delta_des_og = action.moveaxis(-1, 0)

        v_actual = v
        delta_actual = delta
        sign_friction = v.sign()
        throttle_net = self.actuator_model[1] * throttle_og - self.actuator_model[2] * v - self.actuator_model[3] * sign_friction * pitch.cos() - self.actuator_model[5] * g * pitch.sin()

        delta_des = self.actuator_model[0] * (delta_des_og - delta)
        center_mask = return_to_center(delta,delta_des_og)
        delta_des[center_mask] *= self.actuator_model[4]
        delta_des = delta_des.clip(-self.steer_rate_lim, self.steer_rate_lim)

        xd = v_actual * theta.cos() * pitch.cos()
        yd = v_actual * theta.sin() * pitch.cos()
        thd = v * delta_actual.tan() / self.L

#        if thd.max() > 1e5:
#            import pdb;pdb.set_trace()

        vd = throttle_net
        deltad = delta_des
        below_steer_lim = (delta < self.steer_lim[0]).float()
        above_steer_lim = (delta > self.steer_lim[1]).float()
        deltad = deltad.clip(0., 1e3) * below_steer_lim + deltad * (1.-below_steer_lim)
        deltad = deltad.clip(-1e3, 0) * above_steer_lim + deltad * (1.-above_steer_lim)

#        if self.requires_grad:
#            deltad.requires_grad_()
#            deltad.register_hook(lambda grad: grad.clamp(max=1e9))        

        pitchd = torch.zeros_like(pitch)

        return torch.stack([xd, yd, thd, vd, deltad, pitchd], dim=-1)
    
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
            if False:
                self.debug = True
                import pdb;pdb.set_trace()
            else:
                self.debug = False
            action = actions[..., t, :]
            next_state = self.predict(curr_state, action)
#            next_state_clipped = next_state.clone()

#            if self.config['common']['vel_clip']['active']:
#                vel_clip = torch.clamp(next_state[..., [3]], min=self.config['common']
#                    ['vel_clip']['min'], max=self.config['common']['vel_clip']['max'])
#            else:
#                vel_clip = next_state[..., [3]]

#            vel_clip = next_state[..., [3]]

#            theta_mod_up = next_state[..., [2]]
#            converted_angle = torch.atan2(torch.sin(theta_mod_up), torch.cos(theta_mod_up))
#            theta_mod_up_transformed = converted_angle - 2 * torch.pi *(converted_angle > torch.pi)
            
#            assert (theta_mod_up_transformed<=np.pi).all() and (theta_mod_up_transformed>=-np.pi).all()

#            next_state_clipped[...,[2]] = theta_mod_up_transformed
#            next_state_clipped[...,[3]] = vel_clip

#            next_state = next_state_clipped
            X.append(next_state)
            curr_state = next_state.clone()
            
        X_tensor = torch.stack(X,dim=-2)
      
        return X_tensor

    def quat_to_yaw(self, q):
        #quats are x,y,z,w
        qx, qy, qz, qw = q.moveaxis(-1, 0)
        return torch.atan2(2 * (qw*qz + qx*qy), 1 - 2 * (qy*qy + qz*qz))

    def get_observations(self, batch):
        state = batch[self.state_key]

        if len(state.shape) == 1:
            return self.get_observations(dict_map(batch, lambda x:x.unsqueeze(0))).squeeze()

        x = state[..., 0]
        y = state[..., 1]
        q = state[..., 3:7]
        yaw = self.quat_to_yaw(q)
        v = torch.linalg.norm(state[...,7:10], axis=-1)

        actual_direction = torch.arctan2(state[...,8], state[...,7])
#        rev_mask = shortest_distance_bw_angles(actual_direction, yaw).abs() > np.pi/2
#        v[rev_mask] *=-1
        delta = batch[self.steer_key][..., 0] * (30./415.) * (-np.pi/180.)

        pitch = -batch[self.pitch_key][... , 0] #want uphill -> positive pitch

        return torch.stack([x, y, yaw, v, delta, pitch], axis=-1)

    def get_actions(self, batch):
        return batch[...,[0,-1]]

    def to(self, device):
        self.device = device
        return self

if __name__ == '__main__':
    # Sam TODO: write up a test script w/ varying pitches
    import yaml
    import matplotlib.pyplot as plt
    from torch_mpc.setup_mpc import setup_mpc

    config_fp = '/home/atv/physics_atv_ws/src/control/torch_mpc/configs/test_throttle_config.yaml'
    config = yaml.safe_load(open(config_fp, 'r'))
    mpc = setup_mpc(config)
    model = mpc.model

    x0 = torch.zeros(6)
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
