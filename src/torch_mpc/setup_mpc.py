import torch
import numpy as np
import yaml

# models
# from torch_mpc.models.steer_setpoint_kbm import SteerSetpointKBM
# from torch_mpc.models.steer_setpoint_throttle_kbm import SteerSetpointThrottleKBM
# from torch_mpc.models.gravity_throttle_kbm import GravityThrottleKBM
# from torch_mpc.models.actuator_delay import ActuatorDelay
from torch_mpc.models.kbm import KBM

# sampling strategies
from torch_mpc.action_sampling.sampling_strategies.action_library import ActionLibrary
from torch_mpc.action_sampling.sampling_strategies.uniform_gaussian import UniformGaussian
from torch_mpc.action_sampling.sampling_strategies.gaussian_walk import GaussianWalk
from torch_mpc.action_sampling.action_sampler import ActionSampler

# cost functions
from torch_mpc.cost_functions.generic_cost_function import CostFunction
from torch_mpc.cost_functions.cost_terms.footprint_costmap_projection import FootprintCostmapProjection
# from torch_mpc.cost_functions.cost_terms.unknown_map_projection import UnknownMapProjection
from torch_mpc.cost_functions.cost_terms.euclidean_distance_to_goal import EuclideanDistanceToGoal
# from torch_mpc.cost_functions.cost_terms.speed_limit import SpeedLimit
# from torch_mpc.cost_functions.cost_terms.footprint_speedmap_projection import FootprintSpeedmapProjection
# from torch_mpc.cost_functions.cost_terms.valuemap_projection import ValuemapProjection
# from torch_mpc.cost_functions.cost_terms.goal_constraint import GoalConstraint

# update rules
from torch_mpc.update_rules.mppi import MPPI

# mpc
from torch_mpc.algos.batch_sampling_mpc import BatchSamplingMPC

def setup_mpc(config):
    """
    Call this function to set up an MPC instance from the config yaml
    """
    B = config['common']['B']
    H = config['common']['H']
    M = config['common']['M']
    dt = config['common']['dt']
    device = config['common']['device']

    ## setup model ##
    if config['model']['type'] == 'KBM':
        L = config['model']['args']['L']
        min_throttle = config['model']['args']['throttle_lim'][0]
        max_throttle = config['model']['args']['throttle_lim'][1]
        model = KBM(L=L, min_throttle=min_throttle, max_throttle=max_throttle, dt=dt, device=device)

    # if config['model']['type'] == 'SteerSetpointThrottleKBM':
    #     model = SteerSetpointThrottleKBM(**config['model']['args'], dt=dt)
    # elif config['model']['type'] == 'SteerSetpointKBM':
    #     model = SteerSetpointKBM(**config['model']['args'], dt=dt)
    # elif config['model']['type'] == 'GravityThrottleKBM':
    #     model = GravityThrottleKBM(**config['model']['args'], dt=dt)
    else:
        print('Unsupported model type {}'.format(config['model']['type']))
        exit(1)

    # if 'actuator_delay' in config['model'].keys():
    #     model = ActuatorDelay(model=model, buf_len=config['model']['actuator_delay'])

    ## setup sampler ##
    sampling_strategies = {}
    for sv in config['sampling_strategies']['strategies']:
        if sv['type'] == 'ActionLibrary':
            sampling_strategies[sv['label']] = ActionLibrary( #not even used
            B=B, H=H, M=M, device=device, **sv['args'])
        elif sv['type'] == 'UniformGaussian':
            sampling_strategies[sv['label']] = UniformGaussian(
            B=B, H=H, M=M, device=device, **sv['args'])
        elif sv['type'] == 'GaussianWalk':
            sampling_strategies[sv['label']] = GaussianWalk(
            B=B, H=H, M=M, device=device, **sv['args'])
        else:
            print('Unsupported sampling_strategy type {}'.format(sv['type']))
            exit(1)

    action_sampler = ActionSampler(sampling_strategies)

    ## setup cost function ##
    cost_function_params = config['cost_function']
    terms = []
    for term in cost_function_params['terms']:
        params = term['args'] if term['args'] is not None else {}

        if term['type'] == 'EuclideanDistanceToGoal':
            terms.append((
                term['weight'],
                EuclideanDistanceToGoal(**params)
            ))
        elif term['type'] == 'FootprintCostmapProjection':
            terms.append((
                term['weight'],
                FootprintCostmapProjection(**params)
            ))
        # elif term['type'] == 'SpeedLimit':
        #     terms.append((
        #         term['weight'],
        #         SpeedLimit(**params)
        #     ))
        # elif term['type'] == 'FootprintSpeedmapProjection':
        #     terms.append((
        #         term['weight'],
        #         FootprintSpeedmapProjection(**params)
        #     ))
        # elif term['type'] == 'ValuemapProjection':
        #     terms.append((
        #         term['weight'],
        #         ValuemapProjection(**params)
        #     ))
        # elif term['type'] == 'GoalConstraint':
        #     terms.append((
        #         term['weight'],
        #         GoalConstraint(**params)
        #     ))
        else:
            print('Unsupported cost term type {}'.format(term['type']))
            exit(1)
    
    cost_fn = CostFunction(
        terms
    ).to(device)

    ## setup update rules ##
    if config['update_rule']['type'] == 'MPPI':
        update_rule = MPPI(**config['update_rule']['args'])
    else:
        print('Unsupported update_rule type {}'.format(config['update_rule']['type']))
        exit(1)

    ## setup algo ##
    algo = BatchSamplingMPC(model=model, cost_fn=cost_fn, action_sampler=action_sampler, update_rule=update_rule)

    return algo

if __name__ == '__main__':
    config_fp = '/home/atv/physics_atv_ws/src/control/torch_mpc/configs/test_config.yaml'
    config = yaml.safe_load(open(config_fp, 'r'))
    print(config)

    import pdb;pdb.set_trace()
    mpc = setup_mpc(config)
    print(mpc)
