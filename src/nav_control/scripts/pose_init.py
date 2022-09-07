#!/usr/bin/env python
'''
Author: fengsc
Date: 2022-07-26 16:25:37
LastEditTime: 2022-07-31 18:36:38
'''
from time import sleep
import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Empty
import numpy as np

def callback(msg):
    pose = msg.data
    send_msg = PoseWithCovarianceStamped()
    send_msg.header.frame_id = 'map'
    send_msg.header.stamp = rospy.Time.now()
    send_msg.pose.pose.position.x = pose[0]
    send_msg.pose.pose.position.y = pose[1]
    send_msg.pose.pose.position.z = pose[2]
    send_msg.pose.pose.orientation.w = 1
    cov=[0.2,0.2,0,0,0,0]#添加初始方差替代全局定位方案
    send_msg.pose.covariance=np.diag(cov).flatten()
    pose_pub.publish(send_msg)  # 发送预估点
    rospy.sleep(1)

    # rospy.wait_for_service('global_localization')  # 调用全局定位服务得到较精准位置
    # try:
    #     s = rospy.ServiceProxy('global_localization', Empty)
    #     s.call()
    #     s.close()
    # except rospy.ServiceException:
    #     rospy.logwarn("Service 'global_localization' call failed")
    twist_msg = Twist()
    # 移动更新位姿
    for i in range(0, 1000):
        twist_msg.angular.z = 0.8
        twist_pub.publish(twist_msg)
        rospy.sleep(0.01)

    twist_msg.angular.z = 0

    twist_pub.publish(twist_msg)

    # rospy.loginfo("send init_pose: "+str(send_msg.pose.pose.position.x)+" "+str(send_msg.pose.pose.position.y)+" "+str(send_msg.pose.pose.position.z))


def main():
    rospy.init_node("pose_init", anonymous=False)
    rospy.Subscriber("/init_pose", Float32MultiArray, callback, queue_size=1)
    rospy.loginfo("pose_init node initialized!")
    rospy.spin()


if __name__ == "__main__":
    global pose_pub, twist_pub
    pose_pub = rospy.Publisher(
        '/initialpose', PoseWithCovarianceStamped, queue_size=1)
    twist_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    main()
