import torch

class CostFunction:
    """
    High level cost-term aggregator that all MPC should be using to get the costs of trajectories

    In order to handle constraints properly, all cost terms must produce a feasible flag, in
        addition to a cost. We can then check if all trajs are infeasible and do something about it.

    Note that this feasible matrix is meant to be equivalent to allowing cost terms to prune out trajectories,
    but maintaining the batch shape.
    """
    def __init__(self, cost_terms, device='cpu'):
        """
        Args:
            cost_terms: a list of (weight, term) tuples that comprise the cost function
            data_timeout: time after which 
        """
        self.cost_terms = [x[1] for x in cost_terms]
        self.cost_weights = [x[0] for x in cost_terms]
        self.data = {k:None for k in self.get_data_keys()}

        self.device = device

    def cost(self, states, actions):
        """
        Produce costs from states, actions
        Args:
            states: a [B1 x B2 x T x N] tensor of states
            actions: a [B x B2 x T x M] tensor of actions
        """
        costs = torch.zeros(*states.shape[:2]).to(states.device)
        feasible = torch.ones(*states.shape[:2], dtype=bool).to(states.device)

        for cterm, cweight in zip(self.cost_terms, self.cost_weights):
            new_cost, new_feasible = cterm.cost(states, actions, feasible, self.data)

            # new_cost.to(self.device)
            # new_feasible.to(self.device)
            # costs.to(self.device)
            # feasible.to(self.device)
            
            costs += cweight * new_cost
            feasible = feasible & new_feasible

        return costs, feasible

    def get_data_keys(self):
        keys = set()
        for cterm in self.cost_terms:
            for label in cterm.get_data_keys():
                keys.add(label)

        return keys

    def can_compute_cost(self):
        return all([v is not None for v in self.data.values()])

    def to(self, device):
        self.device = device
        self.cost_terms = [x.to(device) for x in self.cost_terms]
        return self

    def __repr__(self):
        out = "Cost Function containing:\n"
        for cterm, cweight in zip(self.cost_terms, self.cost_weights):
            out += "\t{:.2f}:\t{}\n".format(cweight, cterm)
        return out

if __name__ == '__main__':
    cfn = CostFunction(
        cost_terms=[
            (2.0, EuclideanDistanceToGoal()),
            (1.0, CostmapProjection())
        ]
    )

    print(cfn.can_compute_cost())

    cfn.data['goals'] = [
        torch.tensor([
            [5.0, 0.0],
            [10.0, 0.0]
        ]),
        torch.tensor([
            [3.0, 0.0],
            [4.0, 0.0]
        ]),
        torch.tensor([
            [6.0, 0.0],
            [4.0, 0.0]
        ])
    ]

    print(cfn.can_compute_cost())

    cfn.data['costmap'] = torch.zeros(3, 100, 100)
    cfn.data['costmap'][:, 40:60, 60:] = 10.
    cfn.data['costmap_metadata'] = {
        'resolution':torch.tensor([1.0, 0.5, 2.0]),
        'width': torch.tensor([100., 50., 200.]),
        'height': torch.tensor([100., 50., 200.]),
        'origin': torch.tensor([[-50., -50.], [-25., -25.], [-100., -100.]])
    }

    states = torch.zeros(3, 4, 100, 5)
    states[:, 0, :, 0] = torch.linspace(0., 60., 100)
    actions = torch.zeros(3, 4, 100, 2)

    print(cfn.cost(states, actions))
