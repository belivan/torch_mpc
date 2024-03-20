import torch

class ActuatorDelay:
    """
    Model the latency of sending controls to the system as a wrapper around models

    Note that only the rollout function is changed, i.e. single-step model calls don't
    model the delay (as we don't know what timestep we're calling)
    """
    def __init__(self, model, buf_len):
        """
        Args:
            model: the model to but actuator delay on top of
            buf_len: list of buffer lengths (one for each actuator)
        """
        assert len(buf_len) == len(model.action_space().low), "number of buffers != model action space"
        self.model = model

        self.u_lb = self.model.u_lb
        self.u_ub = self.model.u_ub
        self.dt = self.model.dt

        self.buf_len = buf_len
        self.buffers = self.setup_buffers()

    def setup_buffers(self):
        """
        Buffer is M lists w/ t elements
        """
        buffers = []
        for bl in self.buf_len:
            buffers.append(torch.zeros(bl, device=self.model.device))

        return buffers

    def add_to_buffer(self, action):
        """
        Add an actuator command to the front of the buffer
        Args:
            action: the action to add
        """
        for ai, act in enumerate(action):
            self.buffers[ai] = torch.cat([self.buffers[ai][1:], act.view(1)])

    def rollout(self, state, action):
        #copy is necessary but likely bad
        action_modified = action.clone()
        for ai in range(len(self.buffers)):
            buf = self.buffers[ai].to(action.device)
            n = len(buf)
            if n > 0:
                action_modified[..., ai] = torch.cat([
                    buf.view(1, 1, n).repeat(action.shape[0], action.shape[1], 1),
                    action_modified[:, :, :-n, ai]
                ], dim=2)

        return self.model.rollout(state, action_modified)

    #note that this is just a passthrough
    def predict(self, state, action):
        return self.model.predict(state, action)

    def observation_space(self):
        return self.model.observation_space()

    def action_space(self):
        return self.model.action_space()

    def get_observations(self, batch):
        return self.model.get_observations(batch)

    def get_actions(self, batch):
        return self.model.get_actions(batch)
