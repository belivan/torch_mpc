import torch
import numpy as np
import matplotlib.pyplot as plt

class MPPI:
    '''MPPI controller based on Algorithm 2 in [1].
    
    [1] Williams, Grady, et al. "Information theoretic MPC for model-based reinforcement learning." 2017 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2017.

    Args:
        model:
            Model instance. Look at model.py to see interface definition.
        cost_fn:
            Cost function instance. Look at cost.py to see interface definition.
        num_samples:
            Number of samples that MPPI will evaluate per timestep.
        num_timesteps:
            Horizon of control for MPPI controller.
        control_params:
            Dictionary of control parameters defined as follows:
            {
                sys_noise: array of shape [n], where n is the dimension of the model state. Scales the control noise by these values
                temperature: scalar sharpness parameter that scales the weighting of the rollouts
                use: boolean. If false, use gaussian noise, else use ou noise
                ou_alpha: float in [0, 1] to determine the low-pass filter on ou noise
                ou_scale: float to scale the change in rate for ou noise
            }
        num_uniform_samples:
            Number of samples to sample from uniform distribution (to help with changin homotopy classes)
        num_safe_timesteps:
            For safety, set the last k timesteps to have zero throttle (note that this assumes that the first dim of action space is some kind of throttle)

    Batching convention: [batch x time x features]
    '''
    def __init__(self, model, cost_fn, num_samples, num_timesteps, control_params, num_uniform_samples=0, num_safe_timesteps=0, device='cpu'):
        self.model = model
        self.cost_fn = cost_fn
        self.K = num_samples + num_uniform_samples
        self.K1 = num_samples
        self.K2 = num_uniform_samples
        self.T = num_timesteps
        self.sT = num_safe_timesteps
        assert self.sT < self.T, 'expect number of safe timesteps to be less than the horizon, but got ({} > {})'.format(self.sT, self.T)

        self.n = self.model.observation_space().shape[0] #Note that this won't work for high-dim obses
        self.m = self.model.action_space().shape[0]

        self.umin = torch.tensor(self.model.action_space().low)
        self.umax = torch.tensor(self.model.action_space().high)

        self.last_controls = torch.stack([torch.zeros(self.m) for t in range(self.T)], dim=0)
        self.last_states = torch.zeros(self.T, self.n)

        self.noisy_controls = torch.zeros(self.K, self.T, self.m)
        self.noisy_states = torch.zeros(self.K, self.T, self.n)
        self.costs = torch.randn(self.K)
        self.last_cost = torch.randn(size=(1,))
        self.last_weights = torch.zeros(self.K)

        self.sys_noise = torch.tensor(control_params['sys_noise'])  # Equivalent to sigma in [1]
        self.temperature = control_params['temperature']  # Equivalent to lambda in [1]
        self.use_ou = control_params['use_ou']
        self.ou_alpha = control_params['ou_alpha'] if 'ou_alpha' in control_params.keys() else None
        self.ou_scale = control_params['ou_scale'] if 'ou_scale' in control_params.keys() else None
        self.d_ou_scale = control_params['d_ou_scale'] if 'd_ou_scale' in control_params.keys() else None

        self.device = device

    def reset(self):
        self.last_controls = torch.stack([torch.zeros(self.m, device=self.device) for t in range(self.T)], dim=0)
        self.last_states = torch.zeros(self.T, self.n, device=self.device)

        self.noisy_controls = torch.zeros(self.K, self.T, self.m, device=self.device)
        self.noisy_states = torch.zeros(self.K, self.T, self.n, device=self.device)
        self.costs = torch.randn(self.K, device=self.device)
        self.last_cost = torch.randn(size=(1,), device=self.device)
        self.last_weights = torch.zeros(self.K, device=self.device)

    def get_control(self, x, step=True):
        '''Returns action u at current state x

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
        '''
        ## 1. Simulate K rollouts and get costs for each rollout
        sys_noise_unsqueeze = self.sys_noise.view(1, 1, self.m)
        controls_unsqueeze = self.last_controls.view(-1, *self.last_controls.shape)

        du_max = self.umax.view(1, 1, self.m) - controls_unsqueeze
        du_min = self.umin.view(1, 1, self.m) - controls_unsqueeze

        if self.use_ou:
            control_noise = [torch.zeros(size=(self.K1, 1, self.m), device=self.device)]
            dz = torch.randn(size=(self.K1, 1, self.m), device=self.device) * sys_noise_unsqueeze * (self.ou_scale/self.T)
            ddz = torch.randn(size=(self.K1, self.T-1, self.m), device=self.device) * sys_noise_unsqueeze * (self.d_ou_scale/self.T)

            for t in range(self.T-1):
                dz = self.ou_alpha * dz + (1 - self.ou_alpha) * ddz[:, [t]]
                control_noise.append(torch.min(torch.max(control_noise[-1] + dz, du_min[:, [t+1]]), du_max[:, [t+1]]))

            control_noise = torch.cat(control_noise, dim=-2)

        else:
            noise = torch.randn(size=(self.K1, self.T, self.m), device=self.device)
            control_noise = noise * sys_noise_unsqueeze
            control_noise = torch.min(torch.max(control_noise, du_min), du_max)

        #add uniform samples
        if self.K2 > 0:
            unif_noise = du_min + torch.rand(self.K2, self.T, self.m, device=self.device) * (du_max - du_min)

            control_noise = torch.cat([control_noise, unif_noise], axis=0)

        """
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        for k in range(self.K):
            axs[0].plot(control_noise[k, :, 0])
            axs[1].plot(control_noise[k, :, 1])
        plt.show()
        """

        noisy_controls = controls_unsqueeze + control_noise

        #add safety
        if self.sT > 0:
            noisy_controls[:, -self.sT:, 0] = 0.

        states_unsqueeze = torch.stack([x] * self.K, dim=0)

        #Equivalent to (lam * u^T Sig v) for diagonal gaussian Sig
        control_costs = self.temperature * (controls_unsqueeze * control_noise * (1./sys_noise_unsqueeze)).sum(dim=-1).sum(dim=-1)

        trajs = self.model.rollout(states_unsqueeze, noisy_controls)
        costs = self.cost_fn.cost(trajs, noisy_controls) + control_costs

        ## 2. Get minimum cost and obtain normalization constant
        beta = torch.min(costs)
        eta  = torch.sum(torch.exp(-1/self.temperature*(costs - beta)))

        ## 3. Get importance sampling weight
        sampling_weights = (1./eta) * ((-1./self.temperature) * (costs - beta)).exp()

        ## 4. Get action sequence using weighted average
        self.last_controls = (noisy_controls * sampling_weights.view(*sampling_weights.shape, 1, 1)).sum(dim=0)

        ## 5. Return first action in sequence and update self.last_controls
        self.noisy_states = trajs
        self.noisy_controls = noisy_controls
        self.last_states = self.model.rollout(x, self.last_controls)
        self.last_cost = self.cost_fn.cost(self.last_states, self.last_controls)
        self.costs = costs
        self.last_weights = sampling_weights

        u = self.last_controls[0]
        if step:
            self.step()

        return u
        
    def step(self):
        '''Updates self.last_controls to warm start next action sequence.'''
        # Shift controls by one
        self.last_controls = self.last_controls[1:]
        # Initialize last control to be the same as the last in the sequence
        self.last_controls = torch.cat([self.last_controls, self.last_controls[[-1]]], dim=0)
#        self.last_controls = torch.cat([self.last_controls, torch.zeros_like(self.last_controls[[-1]])], dim=0)

    def to(self, device):
        self.device = device
        self.umin = self.umin.to(device)
        self.umax = self.umax.to(device)
        self.last_controls = self.last_controls.to(device)
        self.last_states = self.last_states.to(device)
        self.noisy_controls = self.noisy_controls.to(device)
        self.noisy_states = self.noisy_states.to(device)
        self.costs = self.costs.to(device)
        self.last_cost = self.last_cost.to(device)
        self.last_weights = self.last_weights.to(device)
        self.sys_noise = self.sys_noise.to(device)

        self.model = self.model.to(device)
        self.cost_fn = self.cost_fn.to(device)
        return self

    def viz():
        pass

if __name__ == "__main__":
    from torch_mpc.models.skid_steer import SkidSteer
    from torch_mpc.models.steer_setpoint_kbm import SteerSetpointKBM

    import time

    class TestCost:
        def __init__(self):
            self.device = 'cpu'
        def cost(self, traj, controls):
            stage_cost = self.stage_cost(traj, controls)
            term_cost = self.term_cost(traj[..., -1, :])
            return stage_cost.sum(dim=-1) + term_cost
        def stage_cost(self, traj, controls):
#            state_cost = (traj[...,1] + (traj[...,2]+np.pi/2).abs()) * 10 #drive fast in -y

#            state_cost = ((traj[...,0] - 10.).abs() + (traj[...,1] - 20.)

            state_cost = (traj[...,0]) + 10. * (traj[...,2] - np.pi).abs() #drive fast in -x

#            control_cost = (controls.pow(2) + torch.tensor([[1.0, 100.0]], device=self.device)).sum(dim=-1) #Dont be jerky
            control_cost = torch.zeros_like(state_cost)
            return state_cost + control_cost
        def term_cost(self, tstates):
            return (tstates[..., 1]) * 0.

        def to(self, device):
            self.device = device
            return self

    device = 'cuda'

    mppi_params = {
        'sys_noise':torch.tensor([2.5, 0.15]),
        'temperature':1.0,
        'use_ou':True,
        'ou_alpha':0.9,
        'ou_scale':10.0,
        'd_ou_scale':5.0,
    }

    kbm = SteerSetpointKBM(L=3.0, v_target_lim=[1.0, 5.0], steer_lim=[-0.52, 0.52], steer_rate_lim=0.2, dt=0.15).to(device)

    cfn = TestCost().to(device)
    mppi = MPPI(model=kbm, cost_fn=cfn, num_samples=1024, num_timesteps=50, control_params=mppi_params, num_safe_timesteps=20).to(device)
    x = torch.zeros(kbm.observation_space().shape[0]).to(device)

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

    X = torch.stack(X, dim=0).cpu()
    U = torch.stack(U, dim=0).cpu()

    traj = kbm.rollout(x, mppi.last_controls).cpu()

    print('TRAJ COST = {:.6f}'.format(cfn.cost(X, U)))

    du = abs(U[1:] - U[:-1])
    print('SMOOOTHNESS = {:.6f}'.format(du.mean()))

    t3 = time.time()
    for i in range(100):
        u = mppi.get_control(x)
    t4 = time.time()
    print('ITR TIME = {:.6f}s'.format((t4 - t3)/100.))

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    axs[0].set_title("Traj")
    axs[0].plot(X[:, 0], X[:, 1], c='b', label='traj')
    axs[0].plot(traj[:, 0], traj[:, 1], c='r', label='pred')
    axs[1].set_title("Controls")
    axs[1].plot(U[:, 0],label='throttle')
    axs[1].plot(U[:, 1],label='steer')
    axs[2].set_title("States")
    for xi, name in enumerate(['X', 'Y', 'Th', 'V', 'W']):
        axs[2].plot(X[:,xi], label=name)
    axs[2].legend()
    axs[0].legend()
    axs[1].legend()
    axs[0].set_aspect('equal')
    plt.show()
