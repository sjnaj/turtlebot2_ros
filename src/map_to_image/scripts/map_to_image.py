#!/usr/bin/env python

'''
Author: fengsc
Date: 2022-07-18 22:09:30
LastEditTime: 2022-07-26 20:37:14
'''

import rospy
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import CompressedImage
from nav_msgs.msg import Odometry
import numpy as np
import cv2
import matplotlib.pyplot as plt 
import sys


def callback(occupancyGrid):
    l,w=occupancyGrid.info.width,occupancyGrid.info.height
    # rospy.loginfo(w)
    # length=len(occupancyGrid.data)
    # rospy.loginfo(length);

    # for i in range(0,length):
    #     if occupancyGrid.data[i]==-1:
    #         imagedata[i]=127
    #     elif occupancyGrid.data[i]==0:
    #         imagedata[i]=255
    #     elif occupancyGrid.data[i]==100:
    #         imagedata[i]=0
    #     else:
    #         imagedata[i]=occupancyGrid.data[i]
    imagedata=np.asarray(occupancyGrid.data).reshape(w,l).astype(np.uint8)
    imagedata[imagedata==-1]=255
    imagedata[imagedata==0]=128
    imagedata[imagedata==100]=0
    imagedata= cv2.flip(imagedata, 1, dst=None)#odom与map的x，y坐标相反
    rotate_matrix=cv2.getRotationMatrix2D((w/2,l/2),90,1)#旋转90度
    imagedata= cv2.warpAffine(
    src=imagedata, M=rotate_matrix, dsize=(w, l))
    imagedata=cv2.merge([imagedata,imagedata,imagedata])#1->3
    # imagedata= cv2.cvtColor(imagedata,cv2.COLOR_GRAY2RGBA)
    # cv2.imwrite('2.png', imagedata) # 保存
    success, encoded_image = cv2.imencode(".jpg",imagedata)

    image.data=encoded_image.tobytes()  
    # rospy.loginfo(image.data)
    image.header.stamp = rospy.Time.now() 
  
    image.format = "jpg"
    pub.publish(image)


def main():
    global imagedata,image,pub
    # imagedata = np.zeros(270000,dtype=np.uint8)
    image=CompressedImage()
    
    rospy.init_node('map_to_image', anonymous=False)
    rospy.Subscriber("map", OccupancyGrid, callback,queue_size=1,buff_size=2**30)

    pub = rospy.Publisher('/map/image_raw/compressed',CompressedImage,queue_size=1)

    rospy.loginfo("map_to_image node initialized!")
    
    
    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass