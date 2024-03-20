import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import rospy
class BatchMPPI:
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
        num_safe_timesteps:
            For safe trajectories, command 0 throttle for this many steps at the horizon.
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
            Number of samples to sample from uniform distribution (to help with changing homotopy classes)
        batch_size:
            Number of MPPI optimizations to perform in parallel

    Batching convention: [batch x time x features]
    '''
    def __init__(self, model, cost_fn, num_samples, num_timesteps, control_params, num_uniform_samples=0, batch_size=1, num_safe_timesteps=20, device='cpu',throttle_delay = 0.5):
        self.model = model
        self.cost_fn = cost_fn
        self.K = num_samples + num_uniform_samples
        self.K1 = num_samples
        self.K2 = num_uniform_samples
        self.T = num_timesteps
        self.sT = num_safe_timesteps
        self.B = batch_size

        self.n = self.model.observation_space().shape[0] #Note that this won't work for high-dim obses
        self.m = self.model.action_space().shape[0]

        self.umin = torch.tensor(self.model.action_space().low)
        self.umax = torch.tensor(self.model.action_space().high)

        self.last_states = torch.zeros(self.B, self.T, self.n)
        self.last_controls = torch.zeros(self.B, self.T, self.m)
        self.last_sent_control = torch.zeros(self.B, self.m)

        self.noisy_controls = torch.zeros(self.B, self.K, self.T, self.m)
        self.noisy_states = torch.zeros(self.B, self.K, self.T, self.n)
        self.costs = torch.randn(self.B, self.K)
        self.last_cost = torch.randn(self.B)
        self.last_weights = torch.zeros(self.B, self.K)

        self.sys_noise = torch.tensor(control_params['sys_noise'])  # Equivalent to sigma in [1]
        self.temperature = control_params['temperature']  # Equivalent to lambda in [1]
        self.use_ou = control_params['use_ou']
        self.ou_alpha = torch.tensor(control_params['ou_alpha'],device = device) if 'ou_alpha' in control_params.keys() else None
        self.ou_scale = torch.tensor(control_params['ou_scale'],device = device) if 'ou_scale' in control_params.keys() else None
        self.d_ou_scale = torch.tensor(control_params['d_ou_scale'],device = device) if 'd_ou_scale' in control_params.keys() else None
        self.use_all_last_controls = torch.tensor(control_params['use_all_last_controls']) #[T/F, T/F] throttle, steer
        self.use_normal_init_noise = torch.tensor([int(x) for x in control_params['use_normal_init_noise']]) #[T/F, T/F] throttle, steer
        self.device = device

        self.past_controls = []
        self.throttle_delay = throttle_delay

    def reset(self):
        self.last_controls = torch.zeros(self.B, self.T, self.m, device=self.device)
        self.last_sent_control = torch.zeros(self.B, self.m, device=self.device)
        self.last_states = torch.zeros(self.B, self.T, self.n, device=self.device)

        self.noisy_controls = torch.zeros(self.B, self.K, self.T, self.m, device=self.device)
        self.noisy_states = torch.zeros(self.B, self.K, self.T, self.n, device=self.device)
        self.costs = torch.randn(self.B, self.K, device=self.device)
        self.last_cost = torch.randn(self.B, device=self.device)
        self.last_weights = torch.zeros(self.B, self.K, device=self.device)

    def get_past_throttles(self):
        cur_time = rospy.Time.now().to_sec()
        for i in range(len(self.past_controls)):
            if cur_time - self.past_controls[i][1] <= self.throttle_delay:
                # rospy.loginfo(f"Using past throttle for {len(self.past_controls[i:])}")
                return torch.tensor([x[0] for x in self.past_controls[i:]],device = self.device).reshape((-1,1))
        return None

    def calculate_noisy_controls(self,match_noise):
        sys_noise_unsqueeze = self.sys_noise.view(1, 1, 1, self.m)
        controls_unsqueeze = self.last_controls.view(self.B, 1, self.T, self.m)

        du_max = self.umax.view(1, 1, 1, self.m) - controls_unsqueeze
        du_min = self.umin.view(1, 1, 1, self.m) - controls_unsqueeze

        if self.use_ou:
            if match_noise:
                control_noise = [torch.zeros(size=(self.B, self.K1, 1, self.m), device=self.device)]
                dz = torch.stack([torch.randn(size=(self.K1, 1, self.m), device=self.device)] * self.B, dim=0) * sys_noise_unsqueeze * (self.ou_scale/self.T)
                ddz = torch.stack([torch.randn(size=(self.K1, self.T-1, self.m), device=self.device)] * self.B, dim=0)
            else:
                control_noise = [torch.zeros(size=(self.B, self.K1, 1, self.m), device=self.device)]
                dz = torch.randn(size=(self.B, self.K1, 1, self.m), device=self.device) * sys_noise_unsqueeze * (self.ou_scale/self.T)
                ddz = torch.randn(size=(self.B, self.K1, self.T-1, self.m), device=self.device) * sys_noise_unsqueeze * (self.d_ou_scale/self.T)

            for t in range(self.T-1):
                dz = self.ou_alpha * dz + (1 - self.ou_alpha) * ddz[:, :, [t]]
                # dz = torch.min(torch.max(dz, du_min[:, :, [t+1]]-control_noise[-1]), du_max[:, :, [t+1]]-control_noise[-1])
                control_noise.append(torch.min(torch.max(control_noise[-1] + dz, du_min[:, :, [t+1]]), du_max[:, :, [t+1]]))

            control_noise = torch.cat(control_noise, dim=-2)

        else:
            if match_noise:
                noise = torch.stack([torch.randn(self.K1, self.T, self.m, device=self.device)] * self.B, dim=0)
            else:
                noise = torch.randn(size=(self.B, self.K1, self.T, self.m), device=self.device)

            control_noise = noise * sys_noise_unsqueeze
            control_noise = torch.min(torch.max(control_noise, du_min), du_max)

        """
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        for k in range(self.K1):
            axs[0].plot(control_noise[3, k, :, 0])
            axs[1].plot(control_noise[3, k, :, 1])
        plt.show()
        """

        #add uniform samples
        if self.K2 > 0:
            if match_noise:
                unif_noise = du_min + torch.stack([torch.rand(self.K2, self.T, self.m, device=self.device)] * self.B, dim=0) * (du_max - du_min)
            else:
                unif_noise = du_min + torch.rand(self.B, self.K2, self.T, self.m, device=self.device) * (du_max - du_min)

            control_noise = torch.cat([control_noise, unif_noise], axis=1)

        noisy_controls = controls_unsqueeze + control_noise
        
        past_controls = self.get_past_throttles()        
        if past_controls is not None:
            num_past_control = len(past_controls)
            # import pdb;pdb.set_trace()
            noisy_controls[...,[0]] = torch.cat([past_controls.unsqueeze(dim=0).unsqueeze(dim=0).repeat(1,control_noise.shape[-3],1,1),noisy_controls[...,[0]]], dim=-2)[...,:self.T,:]
        else:
            num_past_control = 0

        if self.sT > 0:
            noisy_controls[:, :, -self.sT:, 0] = 0.
        #Equivalent to (lam * u^T Sig v) for diagonal gaussian Sig
        control_costs = self.temperature * (controls_unsqueeze * control_noise * (1./sys_noise_unsqueeze)).sum(dim=-1).sum(dim=-1)

        return noisy_controls,control_costs,num_past_control    
    
    def rollout_model(self,obs,extra_states,noisy_controls):
        states_unsqueeze = torch.stack([obs] * self.K, dim=1)
        if extra_states is not None:
            extra_state_unsqueezed = torch.stack([extra_states] * self.K, dim=1)
            trajs = self.model.rollout(states_unsqueeze, noisy_controls, extra_states=extra_state_unsqueezed)
        else:
            trajs = self.model.rollout(states_unsqueeze, noisy_controls)
        
        return trajs
    
    def get_control(self, obs, step=True, match_noise=False,extra_states=None):
        '''Returns action u at current state obs

        Args:
            x: 
                Current state. Expects tensor of size self.model.state_dim()

            step:
                optionally, don't step the actions if step=False

            match_noise:
                optional arg to use the same noise distribution for each batch
        
        Returns:
            u: 
                Action from MPPI controller. Returns Tensor(self.m)

            cost:
                Cost from executing the optimal action sequence. Float.

        Batching convention is [B x K x T x {m/n}], where B = batchdim, K=sampledim, T=timedim, m/n = state/action dim
        '''
        ## 1. Simulate K rollouts and get costs for each rollout
        
        noisy_controls,control_costs,num_past_control = self.calculate_noisy_controls(match_noise)
        trajs = self.rollout_model(obs,extra_states,noisy_controls)
        costs = self.cost_fn.cost(trajs, noisy_controls,cur_obs = obs) + control_costs

        ## 2. Get minimum cost and obtain normalization constant
        beta = torch.min(costs, dim=-1, keepdims=True)[0]
        eta  = torch.sum(torch.exp(-1/self.temperature*(costs - beta)), keepdims=True, axis=-1)

        ## 3. Get importance sampling weight
        sampling_weights = (1./eta) * ((-1./self.temperature) * (costs - beta)).exp()

        ## 4. Get action sequence using weighted average
        self.last_controls = (noisy_controls * sampling_weights.view(*sampling_weights.shape, 1, 1)).sum(dim=1)

        ## 5. Return first action in sequence and update self.last_controls
        best_cost_idx = torch.argmin(self.costs, dim=1)
        self.last_states = self.noisy_states[torch.arange(self.B), best_cost_idx]
        self.last_cost = self.costs[torch.arange(self.B), best_cost_idx]
        self.noisy_states = trajs
        self.noisy_controls = noisy_controls
        self.costs = costs
        self.last_weights = sampling_weights

        u = self.last_controls[:, num_past_control]
        self.last_sent_control = u
        if step:
            self.step_throttle_delay(num_past_control)

        return u
    
    def step_throttle_delay(self,num_past_control):
        # if num_past_control > 0:
        self.last_controls = self.last_controls[:, num_past_control+1:]

        self.last_controls = torch.cat([self.last_controls, self.last_controls[:, [-1]].repeat(1,num_past_control+1,1)], dim=1)

            
        
    def step(self):
        '''Updates self.last_controls to warm start next action sequence.'''
        # Shift controls by one
        self.last_controls = self.last_controls[:, 1:]
        # Initialize last control to be the same as the last in the sequence
        self.last_controls = torch.cat([self.last_controls, self.last_controls[:, [-1]]], dim=1)
#        self.last_controls = torch.cat([self.last_controls, torch.zeros_like(self.last_controls[[-1]])], dim=0)

    def to(self, device):
        self.device = device
        self.umin = self.umin.to(device)
        self.umax = self.umax.to(device)
        self.last_controls = self.last_controls.to(device)
        self.last_sent_control = self.last_sent_control.to(device)
        self.use_normal_init_noise  = self.use_normal_init_noise.to(device)
        self.use_all_last_controls = self.use_all_last_controls.to(device)
        self.noisy_controls = self.noisy_controls.to(device)
        self.noisy_states = self.noisy_states.to(device)
        self.costs = self.costs.to(device)
        self.last_weights = self.last_weights.to(device)
        self.sys_noise = self.sys_noise.to(device)
        self.ou_alpha = self.ou_alpha.to(device)
        self.ou_scale = self.ou_scale.to(device)
        self.d_ou_scale = self.d_ou_scale.to(device)

        self.model = self.model.to(device)
        self.cost_fn = self.cost_fn.to(device)
        return self

    def viz():
        pass

if __name__ == "__main__":
    from torch_mpc.models.skid_steer import SkidSteer
    from torch_mpc.models.steer_setpoint_kbm import SteerSetpointKBM
    from torch_mpc.models.clifford_kinematics import CliffordKBM

    import time

    class TestCost:
        def __init__(self):
            self.device = 'cpu'
        def cost(self, traj, controls):
            stage_cost = self.stage_cost(traj, controls)
            term_cost = self.term_cost(traj[..., -1, :])
            return stage_cost.sum(dim=-1) + term_cost
        def stage_cost(self, traj, controls):
#            state_cost = -(traj[...,0] - traj[..., 0, [0]]) * 10 #drive fast in x

#            state_cost += (traj[..., -1, [1]].abs() < 1.).float() * 1e6 #don't end in the middle

#            state_cost = torch.min((traj[...,1] - 1.5).abs(), (traj[..., 1] + 1.5).abs()) * 1000.

            state_cost = (traj[...,1]) + 0. * (traj[...,2] - np.pi).abs() #drive fast in -y

#            control_cost = (controls.pow(2) + torch.tensor([[1.0, 100.0]], device=self.device)).sum(dim=-1) #Dont be jerky
            control_cost = torch.zeros_like(state_cost)
            return state_cost + control_cost
        def term_cost(self, tstates):
            return (tstates[..., 1]) * 0.

        def to(self, device):
            self.device = device
            return self

    device = 'cuda'

    batch_size = 8
    mppi_params = {
#        'sys_noise':torch.tensor([0.1, 0.1, 0.1]),
        'sys_noise':torch.tensor([0.1, 0.1]),
        'temperature':1.0,
        'use_ou':False, 
        'ou_alpha':0.9,
        'ou_scale':10.0,
        'd_ou_scale':5.0,
    }

    kbm = SteerSetpointKBM(L=3.0, v_target_lim=[1.0, 5.0], steer_lim=[-0.52, 0.52], steer_rate_lim=0.2, dt=0.15).to(device)
#    kbm = CliffordKBM(Lf=0.5, Lr=0.5, steer_lim=[-0.52, 0.52], steer_rate_lim=0.2, dt=0.15).to(device)

    cfn = TestCost().to(device)
    mppi = BatchMPPI(batch_size=batch_size, model=kbm, cost_fn=cfn, num_samples=128, num_timesteps=50, control_params=mppi_params).to(device)
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
        axs[1].plot(U[b, :, 1], c='g', label='fsteer' if b == 0 else None)
#        axs[1].plot(U[b, :, 2], c='b', label='rsteer' if b == 0 else None)

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
