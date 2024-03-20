import rospy
import numpy as np

from std_msgs.msg import Float32MultiArray, MultiArrayDimension, Float32;
from grid_map_msgs.msg import GridMap

def costmap_to_gridmap(costmap, msg, costmap_layer='costmap'):
        """
        convert costmap into gridmap msg

        Args:
            costmap: The data to load into the gridmap
            msg: The input msg to extrach metadata from
            costmap: The name of the layer to get costmap from
        """
        costmap_msg = GridMap()
        costmap_msg.info = msg.info
        costmap_msg.layers = ['elevation', costmap_layer]

        costmap_layer_msg = Float32MultiArray()
        costmap_layer_msg.layout.dim.append(
            MultiArrayDimension(
                label="column_index",
                size=costmap.shape[0],
                stride=costmap.shape[0]
            )
        )
        costmap_layer_msg.layout.dim.append(
            MultiArrayDimension(
                label="row_index",
                size=costmap.shape[0],
                stride=costmap.shape[0] * costmap.shape[1]
            )
        )

        costmap_layer_msg.data = costmap[::-1, ::-1].flatten()
        costmap_msg.data.append(costmap_layer_msg)
        costmap_msg.data.append(costmap_layer_msg)
        return costmap_msg

def get_base_msg():
    base_msg = GridMap()
    base_msg.info.length_x = 10.
    base_msg.info.length_y = 10.
    base_msg.info.resolution = 0.05
    base_msg.info.pose.position.x = 5.
    base_msg.info.pose.position.y = 0.
    base_msg.info.pose.position.z = 0.
    base_msg.info.pose.orientation.w = 1.
    base_msg.info.header.frame_id = 'vehicle'
    return base_msg

def get_costmap_msg(base_msg):
    nx = round(base_msg.info.length_x / base_msg.info.resolution)
    ny = round(base_msg.info.length_y / base_msg.info.resolution)

    costmap = np.zeros([nx, ny])
    costmap[70:90, -70:-60] = 10.
    costmap[110:130, -70:-60] = 10.

    costmap[90:110, -10:] = 10.

    costmap_msg = costmap_to_gridmap(costmap, base_msg)
    return costmap_msg

if __name__ == '__main__':
    rospy.init_node('dummy_map_pub')
    rate = rospy.Rate(10.)
    base_msg = get_base_msg()
    costmap_pub = rospy.Publisher('/shortrange_costmap', GridMap, queue_size=1)

    while not rospy.is_shutdown():
        costmap_msg = get_costmap_msg(base_msg)
        costmap_pub.publish(costmap_msg)
        rate.sleep()
