from torch_mpc.models.linear_throttle_kbm import LinearThrottleKBM
from torch_mpc.models.gravity_throttle_kbm import GravityThrottleKBM

str_model = {
    'linear' : LinearThrottleKBM,
    'gravity' : GravityThrottleKBM,
}
