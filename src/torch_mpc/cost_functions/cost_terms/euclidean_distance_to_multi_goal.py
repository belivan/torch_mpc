from torch_mpc.cost_functions.cost_terms.euclidean_distance_to_goal import EuclideanDistanceToGoal
class EuclideanDistanceToMultiGoal(EuclideanDistanceToGoal):
    """
    Simple terminal cost that computes final-state distance to goal
    """
    def __init__(self, prev_cost_thresh,goal_radius, second_goal_penalty, device='cpu'):
        super().__init__(prev_cost_thresh,goal_radius, second_goal_penalty, device)
        self.num_goals = -1

    def __repr__(self):
        return "Euclidean Multi DTG"
