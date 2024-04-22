import scipy
import torch

def dict_map(d1, fn):  // not used
    if isinstance(d1, dict):
        return {k:dict_map(v, fn) for k,v in d1.items()}
    else:
        return fn(d1)

def extract_only_rotation(x,device=None):  // not used
    #return roll,pitch,yaw
    x_squeezed = x.reshape((-1,x.shape[-1])) 
    theta = torch.from_numpy(scipy.spatial.transform.Rotation.from_quat(x_squeezed[..., 3:7].cpu()).as_euler('xyz'))
    theta = theta.reshape((*x.shape[:-1],-1))
    if device is not None:
        return theta.to(device)
    return theta.to(x.device)

def return_to_center(cur,target):
    mask_0 = torch.logical_and(torch.sign(cur) >= 0, torch.sign(target) >= 0)
    mask_1 = torch.logical_and(torch.sign(cur) < 0, torch.sign(target) < 0)
    mask_2 = torch.logical_and(torch.sign(cur) >= 0, torch.sign(target) < 0)
    mask_3 = torch.logical_and(torch.sign(cur) < 0, torch.sign(target) >= 0)

    mask_same = torch.logical_or(mask_0,mask_1)
    mask_diff = torch.logical_or(mask_2,mask_3)

    mask_same_back_to_0 = torch.logical_and(mask_same,torch.abs(cur) > torch.abs(target))

    return torch.logical_or(mask_same_back_to_0,mask_diff)
