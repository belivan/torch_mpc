import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import rospy

class BatchSamplingMPC:
    """
    Implementation of a relatively generic sampling-based MPC algorithm (e.g. CEM, MPPI, etc.)
    We can use this implementation to instantiate a number of algorithms cleanly
    """
    def __init__(self, model, cost_fn, action_sampler, update_rule):
        """
        Args:
            model: The model to optimize over
            cost_fn: The cost function to optimize
            action_sampler: The strategy for sampling actions
            update_rule: The update rule to get best seq from samples
        """
        self.model = model
        self.cost_fn = cost_fn
        self.action_sampler = action_sampler
        self.update_rule = update_rule

        self.device = self.action_sampler.device
        self.B = self.action_sampler.B
        self.H = self.action_sampler.H
        self.K = self.action_sampler.K
        self.n = self.model.observation_space().shape[0] #Note that this won't work for high-dim obses
        self.m = self.model.action_space().shape[0]

        self.setup_variables()

    def setup_variables(self):
        """
        Setup all the variables necessary to run mpc.
        """
        self.last_states = torch.zeros(self.B, self.H, self.n, device=self.device)
        self.last_controls = torch.zeros(self.B, self.H, self.m, device=self.device)

        self.noisy_controls = torch.zeros(self.B, self.K, self.H, self.m, device=self.device)
        self.noisy_states = torch.zeros(self.B, self.K, self.H, self.n, device=self.device)
        self.costs = torch.randn(self.B, self.K, device=self.device)
        self.last_cost = torch.randn(self.B, device=self.device)
        self.last_weights = torch.zeros(self.B, self.K, device=self.device)

        self.u_lb = torch.tensor(self.model.u_lb, device=self.device).view(1, self.m).tile(self.B, 1).float()
        self.u_ub = torch.tensor(self.model.u_ub, device=self.device).view(1, self.m).tile(self.B, 1).float()

    def reset(self):
        self.setup_variables()

    def rollout_model(self,obs,extra_states,noisy_controls):
        states_unsqueeze = torch.stack([obs] * self.K, dim=1)
        if extra_states is not None:
            extra_state_unsqueezed = torch.stack([extra_states] * self.K, dim=1)
            trajs = self.model.rollout(states_unsqueeze, noisy_controls, extra_states=extra_state_unsqueezed)
        else:
            trajs = self.model.rollout(states_unsqueeze, noisy_controls)
        
        return trajs
    
    def get_control(self, obs, step=True, match_noise=False, extra_states=None):
        '''Returns action u at current state obs
        Args:
            x: 
                Current state. Expects tensor of size self.model.state_dim()
            step:
                optionally, don't step the actions if step=False
        Returns:
            u: 
                Action from MPPI controller. Returns Tensor(self.m)
            cost:
                Cost from executing the optimal action sequence. Float.

        Batching convention is [B x K x T x {m/n}], where B = batchdim, K=sampledim, T=timedim, m/n = state/action dim
        '''
        ## Simulate K rollouts and get costs for each rollout
        noisy_controls = self.action_sampler.sample(self.last_controls, self.u_lb, self.u_ub) 
        trajs = self.rollout_model(obs,extra_states,noisy_controls)
        costs, feasible = self.cost_fn.cost(trajs, noisy_controls)

        costs[~feasible] += 1e8 #maybe this is ok?

        optimal_controls, sampling_weights = self.update_rule.update(noisy_controls, costs)

        ## Return first action in sequence and update self.last_controls
        self.last_controls = optimal_controls
        best_cost_idx = torch.argmin(self.costs, dim=1)
        self.last_states = self.noisy_states[torch.arange(self.B), best_cost_idx]
        self.last_cost = self.costs[torch.arange(self.B), best_cost_idx]
        self.noisy_states = trajs
        self.noisy_controls = noisy_controls
        self.costs = costs
        self.last_weights = sampling_weights

        u = self.last_controls[:, 0]

        if step:
            self.step()

        return u, feasible.any(dim=-1)

    def step(self):
        '''Updates self.last_controls to warm start next action sequence.'''
        # Shift controls by one
        self.last_controls = self.last_controls[:, 1:]
        # Initialize last control to be the same as the last in the sequence
        self.last_controls = torch.cat([self.last_controls, self.last_controls[:, [-1]]], dim=1)
#        self.last_controls = torch.cat([self.last_controls, torch.zeros_like(self.last_controls[[-1]])], dim=0)

    def to(self, device):
        self.device = device
        self.u_lb = self.u_lb.to(device)
        self.u_ub = self.u_ub.to(device)
        self.last_controls = self.last_controls.to(device)
        self.noisy_controls = self.noisy_controls.to(device)
        self.noisy_states = self.noisy_states.to(device)
        self.costs = self.costs.to(device)
        self.last_weights = self.last_weights.to(device)

        self.model = self.model.to(device)
        self.cost_fn = self.cost_fn.to(device)
        self.action_sampler = self.action_sampler.to(device)
        return self

    def viz():
        pass

if __name__ == "__main__":
    from torch_mpc.models.skid_steer import SkidSteer
    from torch_mpc.models.steer_setpoint_throttle_kbm import SteerSetpointThrottleKBM
    from torch_mpc.action_sampling.action_sampler import ActionSampler

    import time
    import yaml

    config_fp = '/home/atv/physics_atv_ws/src/control/torch_mpc/configs/test_config.yaml'
    config = yaml.safe_load(open(config_fp, 'r'))

    class TestCost:
        def __init__(self):
            self.device = 'cpu'
        def cost(self, traj, controls):
            stage_cost = self.stage_cost(traj, controls)
            term_cost = self.term_cost(traj[..., -1, :])
            return stage_cost.sum(dim=-1) + term_cost
        def stage_cost(self, traj, controls):
#            state_cost = (traj[...,0] - traj[..., 0, [0]]) * 10 #drive fast in x

#            state_cost += (traj[..., -1, [1]].abs() < 1.).float() * 1e6 #don't end in the middle

#            state_cost = torch.min((traj[...,1] - 1.5).abs(), (traj[..., 1] + 1.5).abs()) * 10.

            state_cost = (traj[...,0]) + 10. * (traj[...,2] - np.pi).abs() #drive fast in -x

#            control_cost = (controls.pow(2) + torch.tensor([[1.0, 100.0]], device=self.device)).sum(dim=-1) #Dont be jerky
            control_cost = torch.zeros_like(state_cost)
            return state_cost + control_cost
        def term_cost(self, tstates):
            return (tstates[..., 1]) * 0.

        def to(self, device):
            self.device = device
            return self


    # TODO: integrate w/ config file
    device = config['common']['device']
    batch_size = config['common']['B']

    kbm = SteerSetpointThrottleKBM(L=3.0, steer_lim=[-0.52, 0.52], steer_rate_lim=0.2, dt=0.1).to(device)
    cfn = TestCost().to(device)
    action_sampler = ActionSampler(config)

    mppi = BatchSamplingMPC(model=kbm, cost_fn=cfn, action_sampler=action_sampler, config=config)
    x = torch.zeros(batch_size, kbm.observation_space().shape[0]).to(device)

    X = []
    U = []

    t0 = time.time()
    for i in range(500):
        X.append(x.clone())
        u = mppi.get_control(x)
        U.append(u.clone())
        x = kbm.predict(x, u)
    t1 = time.time()

    print('TIME = {:.6f}s'.format(t1 - t0))

    X = torch.stack(X, dim=1).cpu()
    U = torch.stack(U, dim=1).cpu()

    traj = kbm.rollout(x, mppi.last_controls).cpu()

    print('TRAJ COST = {}'.format(cfn.cost(X, U)))

    du = abs(U[:, 1:] - U[:, :-1])
    print('SMOOOTHNESS = {}'.format(du.view(batch_size, -1).mean(dim=-1)))

    t3 = time.time()
    for i in range(100):
        u = mppi.get_control(x)
    t4 = time.time()
    print('ITR TIME = {:.6f}s'.format((t4 - t3)/100.))

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    axs[0].set_title("Traj")
    for b in range(batch_size):
        axs[0].plot(X[b, :, 0], X[b, :, 1], c='b', label='traj' if b == 0 else None)
        axs[0].plot(traj[b, :, 0], traj[b, :, 1], c='r', label='pred' if b == 0 else None)
        axs[1].plot(U[b, :, 0], c='r', label='throttle' if b == 0 else None)
        axs[1].plot(U[b, :, 1], c='b', label='steer' if b == 0 else None)

    axs[1].set_title("Controls")
    axs[2].set_title("States")

    colors = 'rgbcmyk'
    for xi, name in enumerate(['X', 'Y', 'Th', 'V', 'W']):
        c = colors[xi % len(colors)]
        for b in range(batch_size):
            axs[2].plot(X[b, :, xi], c=c, label=name if b == 0 else None)

    axs[0].legend()
    axs[1].legend()
    axs[2].legend()

    axs[0].set_aspect('equal')
    plt.show()
