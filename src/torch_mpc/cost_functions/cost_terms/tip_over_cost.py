import torch

from torch_mpc.cost_functions.cost_terms.base import CostTerm
import rospy
class TipOverCost:
    """
    Simple terminal cost that computes final-state distance to goal
    """
    def __init__(self,max_del_steer,max_steer,max_del_steer_vel_range,max_yaw_rate=-1,alpha_del_steer = 30, alpha_yaw_rate = 30,L = 3.0, device='cpu'):
        """
        Args:
            goal_radius: the distance within which a goal is considered reached
            second_goal_penalty: a (high) penalty for states that do not go through the first goal when the first goal is reachable
        """

        self.L = L
        self.alpha_del_steer = torch.clamp(torch.exp(torch.tensor(alpha_del_steer)),max = 1e10)
        self.alpha_yaw_rate = torch.clamp(torch.exp(torch.tensor(alpha_yaw_rate)),max = 1e10)
        self.max_del_steer = max_del_steer
        self.max_steer = max_steer
        self.max_del_steer_vel_range = max_del_steer_vel_range

        self.max_yaw_rate = max_yaw_rate
        self.device = device


    def get_data_keys(self):
        return []

    def cost(self, states, actions, data,past_cost=None,cur_obs = None):
        cost = torch.zeros(states.shape[:2], device=self.device)

        world_speed = states[..., 3].abs()
        future_steer = states[..., 4]

        if isinstance(cur_obs,dict):
            cur_steer = cur_obs['new_state'][...,-1]
        else:
            cur_steer = cur_obs[...,-1]

        init_change_in_steer = (actions[...,0,-1] - cur_steer)

        max_del_steer = torch.zeros_like(init_change_in_steer)
        max_steer = torch.zeros_like(actions[...,-1])
        for i in range(len(self.max_del_steer_vel_range)):
            if i ==0 :
                mask = world_speed < self.max_del_steer_vel_range[i]
            else:
                mask = world_speed < self.max_del_steer_vel_range[i]
                mask = torch.logical_and(mask,world_speed > self.max_del_steer_vel_range[i-1])
            max_del_steer[mask[...,0]] = self.max_del_steer[i]
            max_steer[mask] = self.max_steer[i]
        remaining_mask = max_del_steer == 0
        max_del_steer[remaining_mask] = self.max_del_steer[-1]
        max_steer[remaining_mask] = self.max_steer[-1]

        cost += self.alpha_del_steer * (init_change_in_steer.abs() > max_del_steer)

        if self.max_yaw_rate > 0:
            yaw_rate = world_speed * torch.tan(future_steer) / self.L
            cost += torch.sum(self.alpha_yaw_rate * (yaw_rate.abs() > self.max_yaw_rate),axis=-1)

        cost += torch.sum(self.alpha_del_steer * (states[...,-1].abs() > max_steer),axis=-1)

        return cost

    def to(self, device):
        self.device = device
        return self

    def __repr__(self):
        return "Tip Over Cost"
