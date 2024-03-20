#!/usr/bin/python3
from torch_mpc.scripts.ros.mppi_node_throttle import *
from torch_mpc.algos.batch_mppi_tartan_devel import BatchMPPI as TartanBatchMPPI
from wheeledsim_rl.util.util import *
class TartanMPPINode(MPPINode):
    def __init__(self, model, kbm, cost_fn, mppi, dt, converter, base_frame, mppi_itrs,rgb_list, pitch_window, pitch_thresh,goal_radius):
        super().__init__(kbm, cost_fn, mppi, dt, converter, base_frame, mppi_itrs,rgb_list,pitch_window,pitch_thresh, goal_radius)
        self.model = model
    
    def get_obs_control(self):
        t1 = time.time()
        data = self.converter.get_data()
        obs = dict_to(data['observation'],self.mppi.device)
        obs = dict_map(obs,lambda x: x.unsqueeze(dim=0))
        t2 = time.time()

        for _ in range(self.mppi_itrs-1):
            self.mppi.get_control(obs, step=False)


        result = self.mppi.get_control(obs).cpu()[0]
        t3 = time.time()
        rospy.loginfo_throttle(1.0, 'DATA LOAD: {:.4f}'.format(t2-t1))
        rospy.loginfo_throttle(1.0, 'MPPI: {:.4f}'.format(t3-t2))
        self.mppi_time.append(t3-t2)
        return result , data


if __name__ == '__main__':
    rospy.init_node('mppi_node_tartan')
    run(TartanBatchMPPI, TartanMPPINode)
