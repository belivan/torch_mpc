import torch

from torch_mpc.cost_functions.cost_terms.base import CostTerm

class GoalConstraint:
    """
    Simple check that forces MPPI to go through the goal if any feasible trajectories currently go through the goal
    """
    def __init__(self, goal_radius=2., goal_key='waypoints', device='cpu'):
        """
        Args:
            goal_radius: the distance within which a goal is considered reached
            goal_key: the key in data to get goals from
        """
        self.goal_radius = goal_radius
        self.goal_key = goal_key
        self.device = device

    def get_data_keys(self):
        return [self.goal_key]

    def cost(self, states, actions, feasible, data):
        cost = torch.zeros(states.shape[:2], device=self.device)
        new_feasible = torch.ones(states.shape[:2], dtype=bool, device=self.device)
        
        #for-loop here because the goal array can be ragged
        for bi in range(states.shape[0]):
            bgoals = data[self.goal_key][bi]

            world_pos = states[bi, ..., :2]
            #compute whether any trajs have reached the first goal
            first_goal_dist = torch.linalg.norm(world_pos - bgoals[0], axis=-1)

            traj_reached_goal = torch.any(first_goal_dist < self.goal_radius, axis=-1) & feasible

            #we should optimize past the first goal
            if torch.any(traj_reached_goal):
                print('constraining!')
                new_feasible = traj_reached_goal
            else:
                new_feasible = feasible.clone()

        return cost, new_feasible

    def to(self, device):
        self.device = device
        return self

    def __repr__(self):
        return "Goal Constraint"
