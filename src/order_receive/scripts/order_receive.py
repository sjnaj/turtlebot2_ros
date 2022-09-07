#!/usr/bin/env python
'''
Author: fengsc
Date: 2022-07-23 22:20:28
LastEditTime: 2022-09-05 22:25:16
'''
from matplotlib.pyplot import switch_backend
import rospy
from std_msgs.msg import UInt8
import os,sys
import signal


def kill_process(name):
    # 由用户输入进程名
    try:
        # 通过进程名来过滤ps ax的结果
        for line in os.popen("ps ax | grep " + name + " | grep -v grep"):
            print('line: {}'.format(line))
            fields = line.split()
            print('fields: {}'.format(fields))

            # 解析出pid
            pid = fields[0]
            print('pid: {}'.format(pid))

            # 以发送信号的形式杀掉进程
            os.kill(int(pid), signal.SIGKILL)
        print("Process Successfully terminated")

    except:
        print("Error Encountered while running script")


def save_map():
    global num
    os.system(f"gnome-terminal -e '{shell} -c \"cd {map_dir}; source {workspace_path}/devel/setup.{shell};rosrun map_server map_saver -f mymap{num}; exec {shell}\"'")
    num += 1


def open_map(i):
    kill_process("rviz")#不能多开
    if os.path.isfile(f"{workspace_path}/Map/mymap{i}.yaml"):
        os.system(
            f"gnome-terminal -e '{shell} -c \"cd {workspace_path}; source devel/setup.{shell}; roslaunch turtlebot_navigation rplidar_amcl_demo.launch map_file:={map_dir}/mymap{i}.yaml; exec {shell}\"'")
    

def slam():
    kill_process("rviz")
    os.system(
        f"gnome-terminal -e '{shell} -c \"cd {workspace_path}; source devel/setup.{shell}; roslaunch turtlebot_navigation rplidar_gmapping_demo.launch; exec {shell}\" '")


def callback(msg):

    order = msg.data
    if order == 0:
        slam()
    elif order == 4:
        save_map()
    else:  # 1 2 3 three map
        open_map(order)



global workspace_path
workspace_path =sys.argv[1]
global shell
shell=sys.argv[2]
global map_dir
map_dir =sys.argv[3]
global num#map编号
num = len(os.listdir(map_dir))//2+1


def main():

    rospy.init_node("order_receive", anonymous=False)
    rospy.Subscriber("order", UInt8, callback, queue_size=1)
    rospy.loginfo("order_receive node initialized!")
    rospy.spin()


if __name__ == "__main__":
    main()
