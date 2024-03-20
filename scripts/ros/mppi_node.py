#!/usr/bin/python3

import rospy
import yaml

from torch_mpc.ros_converter.ros_converter import ROSConverter
from torch_mpc.setup_mpc import setup_mpc
from torch_mpc.ros.mppi_node import MPPINode

if __name__ == '__main__':
    rospy.init_node('mppi_node')

    #data params
    config_fp = rospy.get_param("~config_fp")
    base_frame = rospy.get_param("~base_frame")
    mpc_itrs = rospy.get_param("~mpc_itrs")

    config = yaml.safe_load(open(config_fp, 'r'))
    mpc = setup_mpc(config)
    model = mpc.model
    cost_fn = mpc.cost_fn
    dt = config['common']['dt']

    converter = ROSConverter(config['ros'])

    mppi_node = MPPINode(model, cost_fn, mpc, dt, converter, base_frame, mpc_itrs)

    rate = rospy.Rate(1)
    rate.sleep()
    mppi_node.spin()
