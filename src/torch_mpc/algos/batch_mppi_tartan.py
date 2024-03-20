import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from wheeledsim_rl.util.util import *
from torch_mpc.algos.batch_mppi import BatchMPPI as BaseMPPI
from copy import deepcopy
class BatchMPPI(BaseMPPI):
    def __init__(self, model,kbm, cost_fn, num_samples, num_timesteps, control_params, num_uniform_samples=0, batch_size=1, num_safe_timesteps=20, device='cpu',throttle_delay = 0.5):
        super().__init__(kbm, cost_fn, num_samples, num_timesteps, control_params, num_uniform_samples, batch_size, num_safe_timesteps, device,throttle_delay)
        self.model =model

    def rollout_model(self,obs,extra_states,noisy_controls):
        obs_unsqueeze = {}
        for k in obs.keys():
            if k not in ['state','new_state','steer_angle']:
                obs_unsqueeze[k] = torch.stack([obs[k]] * 1, dim=1)
                
            else:
                obs_unsqueeze[k] = torch.stack([obs[k]] * self.K, dim=1)
        self.model.eval()
        tartan_cmd = deepcopy(noisy_controls)
        tartan_cmd[...,1] = tartan_cmd[...,1] / (-0.52)
        with torch.no_grad():
            model_pred = self.model.predict(obs_unsqueeze, tartan_cmd, return_info=False,decode_obs=False,same_init_step = True, batch_size = self.K)['new_state'].mean

        trajs = new_state_to_kbm5dim(model_pred,self.device)
        return trajs

