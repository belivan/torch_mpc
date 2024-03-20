import torch
import numpy as np

from torch_mpc.models.gravity_throttle_kbm import GravityThrottleKBM
from torch_mpc.models.str_to_model import *
from torch_mpc.models.str_to_model import str_model
from torch_mpc.util.utils import *
import matplotlib.pyplot as plt

class MultiVelKBM(GravityThrottleKBM): 

    def __init__(self,L=3.0, throttle_lim=[0., 1.0], steer_lim=[-0.52, 0.52], steer_rate_lim=0.45, dt=0.1, device='cpu', config=None, requires_grad=False, ):
        self.L = L
        self.dt = dt
        self.device = device
        self.throttle_lim = throttle_lim
        

        self.u_lb = np.array([throttle_lim[0], steer_lim[0]])
        self.u_ub = np.array([throttle_lim[1], steer_lim[1]])
        self.x_lb = -np.ones(5).astype(float) * float('inf')
        self.x_ub = np.ones(5).astype(float) * float('inf')

        self.steer_lim = steer_lim
        self.steer_rate_lim = steer_rate_lim

        self.config=config
        self.requires_grad = requires_grad

        self.models = {}
        # self.num_models = len(self.config['models'])
        self.model_type = self.config['common']['model_type']

        self.model_ids = self.config['model_seq']

        
        
        self.reset_parameters()
        
    def reset_parameters(self,requires_grad= None):
        if requires_grad == None:
            requires_grad = self.requires_grad

        self.opt_param_names = []
        self.parameters = {}
        
        for model_id,model_config in self.config['models'].items():
            self.models[model_id] = str_model[self.model_type](self.L,self.throttle_lim,self.steer_lim,self.steer_rate_lim,self.dt,self.device,model_config,requires_grad = requires_grad)
            self.opt_param_names += [model_id + 'x' + k for k in self.models[model_id].opt_param_names]
            for k in self.models[model_id].parameters:
                self.parameters[model_id + 'x' + k] = self.models[model_id].parameters[k]
        colors = plt.cm.jet(np.linspace(0,1,70))
        self.colors = {}
        for i,k in enumerate(self.parameters):
            self.colors[k] = colors[i]
    def opt_to_kbm_param(self, opt_param):
        model_id = opt_param.split('x')[0]
        param_name = opt_param.split('x')[-1]
        return self.models[model_id].parameters[param_name]
    
    def update_parameters(self, params):
        for k in params:
            model_id = k.split('x')[0]
            param_name = k.split('x')[-1]
            # update_dict = {k.split('_')[-1] : params[k]}
            self.models[model_id].parameters[param_name] = params[k]

    
    def dynamics(self,state, extra_state, action):
        if self.debug:
            import pdb;pdb.set_trace()
        x, y, theta, v, delta = state.moveaxis(-1, 0)
        throttle_og, delta_des_og = action.moveaxis(-1, 0)

        v_actual = v
        delta_actual = delta
        sign_friction = torch.sign(v)
        # import pdb;pdb.set_trace()
        sample_model_params = self.models[self.model_ids[0]].parameters
        if not self.config['common']['multi_vel_split_active']:
            params={k:sample_model_params[k] for k in sample_model_params.keys()}
        else:
            params={k:torch.zeros_like(v) for k in sample_model_params}
            mask_total = torch.zeros_like(v)
            assert len(self.config['common']['multi_vel_splits']) == len(self.models)
            for vel_split_idx in range(len(self.config['common']['multi_vel_splits'])-1):
                mask_1 = v.abs() < self.config['common']['multi_vel_splits'][vel_split_idx+1]
                mask_2 = v.abs() >= self.config['common']['multi_vel_splits'][vel_split_idx]
                mask = torch.logical_and(mask_1,mask_2)
                    
                if self.config['common']['continuous']:

                    weight = (v[mask] - self.config['common']['multi_vel_splits'][vel_split_idx])/(self.config['common']['multi_vel_splits'][vel_split_idx+1] - self.config['common']['multi_vel_splits'][vel_split_idx])
                
                    for k in sample_model_params:
                        params[k][mask] = weight * self.models[self.model_ids[vel_split_idx+1]].parameters[k] + (1-weight) * self.models[self.model_ids[vel_split_idx]].parameters[k]
                
                else:
                    for k in sample_model_params:
                        params[k][mask] = self.models[self.model_ids[vel_split_idx]].parameters[k]
                
                mask_total = torch.logical_or(mask_total,mask)
        remaining_mask = torch.logical_not(mask_total)
        for k in sample_model_params:
            params[k][remaining_mask] = self.models[self.model_ids[-1]].parameters[k]
        throttle_net = params['1'].exp() *throttle_og - params['2'].exp() * v - params['3'].exp() * sign_friction
        delta_des = params['0'].exp() * (delta_des_og - delta)
        center_mask = return_to_center(delta,delta_des_og)
        delta_des[center_mask] *= params['4'][center_mask].exp() 
        delta_des = delta_des.clip(-self.steer_rate_lim, self.steer_rate_lim)

        xd = v_actual * theta.cos()
        yd = v_actual * theta.sin()
        thd = v * delta_actual.tan() / self.L
        if thd.max() > 1e5:
            import pdb;pdb.set_trace()
        vd = throttle_net
        deltad = delta_des
        below_steer_lim = (delta < self.steer_lim[0]).float()
        above_steer_lim = (delta > self.steer_lim[1]).float()
        deltad = deltad.clip(0., 1e3) * below_steer_lim + deltad * (1.-below_steer_lim)
        deltad = deltad.clip(-1e3, 0) * above_steer_lim + deltad * (1.-above_steer_lim)
        if self.requires_grad:
            deltad.requires_grad_()
            deltad.register_hook(lambda grad: grad.clamp(max=1e9))		
        return torch.stack([xd, yd, thd, vd, deltad], dim=-1)