#!/usr/bin/env python

import rospy 
import sys
import numpy as np 

from geometry_msgs.msg import Twist

sys.path.append('/home/amber/CamArray/FullStack/cpp_version/py_utils')
from mpac_cmd import * 

u = np.array([0.0,0.0])


def callback(data):
    print(data.linear.x)
    print(data.angular.z)
    u[0] = data.linear.x
    u[1] = data.angular.z

    print(u)
    walk_mpc_idqp(vx = u[0], vrz = u[1])


def commander(): 
    rospy.init_node('commander', anonymous=True)
    rospy.Subscriber("u_star", Twist, callback)

    rospy.spin()


if __name__=='__main__': 
    commander()


