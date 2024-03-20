import torch

def move_to_local_frame(traj, xidx=0, yidx=1, thidx=2):
    """
    Transform a trajectory into its local frame. I.e., translate by -(x, y),
        then rotate by -th. These values will be taken from the initial state
        of each trajectory. Note that this only handles positions (i.e. assume
        velocity is scalar/in body frame)

    Args:
        traj: A [B x K x T x 2] Tensor of trajectories
        xidx: The index of the x coordinate in state
        yidx: The index of the y coordinate in state
        thidx: The index of the th coordinate in state
    """
    ops = traj[..., 0, [xidx, yidx]].view(traj.shape[0], traj.shape[1], 1, 2)
    oths = traj[..., 0, thidx].view(traj.shape[0], traj.shape[1], 1, 1)

    #translate
    pos_out = traj[..., [xidx, yidx]] - ops

    #rotate
    th_out = traj[..., [thidx]] - oths

    R = torch.stack([
        torch.cat([oths.cos(), -oths.sin()], dim=-1),
        torch.cat([oths.sin(), oths.cos()], dim=-1),
    ], dim=-1)

    pos_out = R @ pos_out.view(*pos_out.shape, 1)
    pos_out = pos_out.view(*pos_out.shape[:-1])

    traj_out = traj.clone()
    traj_out[..., xidx] = pos_out[..., 0]
    traj_out[..., yidx] = pos_out[..., 1]
    traj_out[..., thidx] = th_out[..., 0]
    return traj_out

def world_to_grid(world_pos, metadata):
    '''Converts the world position (x,y) into indices that can be used to access the costmap.
    
    Args:
        world_pos:
            Tensor(B, K, T, 2) representing the world position being queried in the costmap.
        metadata:
            res, length, width (B,)
            origin (B, 2)

    Returns:
        grid_pos:
            Tensor(B, K, T, 2) representing the indices in the grid that correspond to the world position
    '''
    res = metadata['resolution']
    nx = (metadata['length_x']/res).long()
    ny = (metadata['length_y']/res).long()
    ox = metadata['origin'][:, 0]
    oy = metadata['origin'][:, 1]

    trailing_dims = [1] * (world_pos.ndim-2)

    gx = (world_pos[..., 0] - ox.view(-1, *trailing_dims)) / res.view(-1, *trailing_dims)
    gy = (world_pos[..., 1] - oy.view(-1, *trailing_dims)) / res.view(-1, *trailing_dims)

    grid_pos = torch.stack([gx, gy], dim=-1).long()
    invalid_mask = (grid_pos[..., 0] < 0) | (grid_pos[..., 1] < 0) | (grid_pos[..., 0] >= nx.view(-1, *trailing_dims)) | (grid_pos[..., 1] >= ny.view(-1, *trailing_dims))

    return grid_pos, invalid_mask

def value_iteration(costmap, metadata, goals, tol=1e-4, gamma=1.0, max_itrs=1000):
    """
    Perform (batched) value iteration on a costmap
    Args:
        costmap: Tensor (B x W x H) costmap to perform value iteration over
        metadata: map metadata
        goals: Tensor (B x 2) of goal positions for each costmap
    """

    #have to assume same size/metadata for each map
    B = costmap.shape[0]
    res = metadata['resolution']
    nx = (metadata['height']/res).long()[0]
    ny = (metadata['width']/res).long()[0]

    ## setup ##
    V = torch.ones(B, nx+2, ny+2, device=costmap.device) * 1e10 #simplest to handle padding here
    R = torch.ones(B, nx+2, ny+2, device=costmap.device) * 1e10
    R[:, 1:-1, 1:-1] = costmap

    ## load in goal point ##
    goal_grid_pos, invalid_mask = world_to_grid(torch.stack([x[0] for x in goals], dim=0).view(B, 1, 1, 2), metadata)
    goal_grid_pos = goal_grid_pos.view(B, 2)[:, [1, 0]]

    R[torch.arange(B), goal_grid_pos[:, 0]+1, goal_grid_pos[:, 1]+1] = 0.

    ## perform value iteration ##
    for i in range(nx + ny):
        Rsa = torch.stack([costmap] * 5, dim=1) #[B x 5 x W x H]

        ## handle terminal state ##
        V[torch.arange(B), goal_grid_pos[:, 0]+1, goal_grid_pos[:, 1]+1] = 0.

        Vs_next = torch.stack([
            V[:, 1:-1, 1:-1], #stay
            V[:, 2:, 1:-1], #up
            V[:, :-2, 1:-1], #down
            V[:, 1:-1, 2:], #left
            V[:, 1:-1, :-2] #right
        ], dim=1) #[B x 5 x W x H]

        Qsa = Rsa + gamma * Vs_next

        Vnext = Qsa.min(dim=1)[0]
        Vnext[torch.arange(B), goal_grid_pos[:, 0], goal_grid_pos[:, 1]] = 0.

        err = (V[:, 1:-1, 1:-1] - Vnext).abs().max()
        if err < tol:
            break

        V[:, 1:-1, 1:-1] = Vnext

    return V[:, 1:-1, 1:-1]
