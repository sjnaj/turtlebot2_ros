#!/usr/bin/env python
from ast import Pass
import rospy
import math

import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatus
from geometry_msgs.msg import Pose, Point, Quaternion
from tf.transformations import quaternion_from_euler
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PoseWithCovarianceStamped

import cv2
import numpy as np


objp = np.zeros((6 * 9, 3), np.float32)
# 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
objp = 2.6 * objp   # 打印棋盘格一格的边长为2.6cm

obj_points = objp   # 存储3D点
mtx = np.asarray([[1.81012669e+03, 0.00000000e+00, 1.19067989e+03],
                  [0.00000000e+00, 1.80847669e+03, 5.31567124e+02],
                  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist = np.asarray(([[3.81200664e-02,  7.49764725e-01, -1.36749937e-03,
                     -2.20345807e-03, -5.91466654e+00]]))
Camera_intrinsic = {"mtx": mtx, "dist": dist, }


class MoveBaseSeq():

    def __init__(self):

        rospy.init_node('target_pub')
        # List of goal poses:
        self.pose_seq = list()

        self.points_sub = rospy.Subscriber(
            'target_poses', Float32MultiArray, self.callback, queue_size=1)
        self.image_sub = rospy.Subscriber(
            'init_image', CompressedImage, self.callback1, queue_size=1)
        self.pose_sub = rospy.Subscriber(
            'amcl_pose', PoseWithCovarianceStamped, self.callback2, queue_size=1)
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)

        self.P_ar_robot = np.array([0, 0, 0])  # ar -> robot
        self.P_ros_robot = np.array([0, 0, 0])

        rospy.loginfo("target_pub node initialized!")

        rospy.spin()

    def active_cb(self):
        rospy.loginfo("Goal pose "+str(self.goal_cnt+1) +
                      " is now being processed by the Action Server...")

    def feedback_cb(self, feedback):
        pass
        # To print current pose at each feedback:
        # rospy.loginfo("Feedback for goal pose " +
        #               str(self.goal_cnt+1)+" received")

    def done_cb(self, status, result):
        self.goal_cnt += 1
    # Reference for terminal status values: http://docs.ros.org/diamondback/api/actionlib_msgs/html/msg/GoalStatus.html
        if status == 2:
            rospy.loginfo("Goal pose "+str(self.goal_cnt) +
                          " received a cancel request after it started executing, completed execution!")

        if status == 3:
            rospy.loginfo("Goal pose "+str(self.goal_cnt)+" reached")
            if self.goal_cnt < len(self.pose_seq):
                next_goal = MoveBaseGoal()
                next_goal.target_pose.header.frame_id = "map"
                next_goal.target_pose.header.stamp = rospy.Time.now()
                next_goal.target_pose.pose = self.pose_seq[self.goal_cnt]
                rospy.loginfo("Sending goal pose " +
                              str(self.goal_cnt+1)+" to Action Server")
                rospy.loginfo(str(self.pose_seq[self.goal_cnt]))
                self.client.send_goal(
                    next_goal, self.done_cb, self.active_cb, self.feedback_cb)
            else:
                rospy.loginfo("Final goal pose reached!")
                rospy.signal_shutdown("Final goal pose reached!")
                return

        if status == 4:
            rospy.loginfo("Goal pose "+str(self.goal_cnt) +
                          " was aborted by the Action Server")
            rospy.signal_shutdown(
                "Goal pose "+str(self.goal_cnt)+" aborted, shutting down!")
            return

        if status == 5:
            rospy.loginfo("Goal pose "+str(self.goal_cnt) +
                          " has been rejected by the Action Server")
            rospy.signal_shutdown(
                "Goal pose "+str(self.goal_cnt)+" rejected, shutting down!")
            return

        if status == 8:
            rospy.loginfo("Goal pose "+str(self.goal_cnt) +
                          " received a cancel request before it started executing, successfully cancelled!")

    def movebase_client(self):
        wait = self.client.wait_for_server(rospy.Duration(5.0))
        if not wait:
            rospy.logerr("Action server not available!")
            rospy.signal_shutdown("Action server not available!")
            return
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose = self.pose_seq[self.goal_cnt]
        rospy.loginfo("Sending goal pose " +
                      str(self.goal_cnt+1)+" to Action Server")
        rospy.loginfo(str(self.pose_seq[self.goal_cnt]))
        self.client.send_goal(goal, self.done_cb,
                              self.active_cb, self.feedback_cb)

    def callback(self, msg):
        self.goal_cnt = 0
        if(msg.data[len(msg.data)-1] == -100):  # AR模式
            points_seq = [self.P_ros_robot-self.P_ar_robot+msg.data[i:i+3]
                          for i in range(0, len(msg.data)-3, 3)]
        else:
            points_seq = [msg.data[i:i+3] for i in range(0, len(msg.data), 3)]

        for point in points_seq:
            point[2] = 0  # 高度为0，不考虑标记物的旋转
            self.pose_seq.append(
                Pose(point, quaternion_from_euler(0, 0, 0, axes='sxyz')))  # 默认姿态
        self.movebase_client()

    def callback1(self, msg):
        nparr = np.fromstring(msg.data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
        if ret:    # 画面中有棋盘格
            img_points = np.array(corners)
            cv2.drawChessboardCorners(frame, (9, 6), corners, ret)
            # rvec: 旋转向量 tvec: 平移向量
            _, rvec, tvec = cv2.solvePnP(
                obj_points, img_points, Camera_intrinsic["mtx"], Camera_intrinsic["dist"])    # 解算位姿
            distance = math.sqrt(tvec[0]**2+tvec[1]**2+tvec[2]**2)  # 计算距离
            print(distance)
            self.P_robot_ar = tvec
            # rvec_matrix = cv2.Rodrigues(rvec)[0]    # 旋转向量->旋转矩阵
            # proj_matrix = np.hstack((rvec_matrix, tvec))    # 合并
            # self.homo_matrix = np.vstack(proj_matrix, np.array([0, 0, 0, 1])) # 齐次矩阵\\\\\\\\\\\\\\\\\\\\\\

            # eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]  # 欧拉角
            # pitch, yaw, roll = eulerAngles[0], eulerAngles[1], eulerAngles[2]
            # cv2.putText(frame, "dist: %.2fcm, yaw: %.2f, pitch: %.2f, roll: %.2f" % (
            #     distance, yaw, pitch, roll), (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # cv2.imshow('frame', frame)
        # else:   # 画面中没有棋盘格
        #     cv2.putText(frame, "Unable to Detect Chessboard", (20,
        #                 frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)
        #     cv2.imshow('frame', frame)
    def callback2(self, msg):
        position = msg.pose.pose.position
        self.P_ros_robot = np.array([position.x, position.y, position.z])


if __name__ == '__main__':
    try:
        MoveBaseSeq()
    except rospy.ROSInterruptException:
        rospy.loginfo("Navigation finished.")
