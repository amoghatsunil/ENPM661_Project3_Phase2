#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import pandas as pd
import numpy as np
import time
# TurtleBot3 constants (adjust if needed)
RADIUS = 0.033     # Wheel radius in meters
LENGTH = 0.160     # Distance between wheels in meters
RPM_TO_RAD = 2 * 3.1416 / 60  # Convert RPM to rad/s

# Canvas and real world map scaling
CANVAS_WIDTH = 1080   # px
CANVAS_HEIGHT = 600   # px
REAL_WIDTH = 5400     # mm
REAL_HEIGHT = 3000    # mm
SCALE_X = REAL_WIDTH / CANVAS_WIDTH  # mm per px
SCALE_Y = REAL_HEIGHT / CANVAS_HEIGHT

df = pd.read_csv("final_path_actions.csv")

class PathExecutor(Node):
    def __init__(self):
        super().__init__('waypoint_nav')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer_period = 1
        self.index = 0
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        self.get_logger().info("Path Executor Node Started.")
        self.actions = [(0,20),(0,20),(20,0),(0,20),(25,25),(25,25),(25,25),(25,25),(20,20),(25,25),(20,20),(25,25),(25,25),(20,20),(25,25),(25,25),(20,20),(25,25),
                        (20,0),(20,0),(25,0),(20,20),(25,25),(20,20),(25,25),(25,0),(25,25),(25,25),(25,25),(25,25),(25,25),(25,25),(20,20),(25,25),(25,25),(25,25),
                        (20,20),(20,20),(20,0),(25,25),(25,25),(25,25),(25,25),(25,25),(0,20),(20,20),(25,25),(25,25),(20,20),(20,20),(0,25),(0,25),(25,25),(25,25),
                        (20,20),(0,25),(25,25),(25,25),(20,20),(25,25),(25,25),(25,25),(25,25),(20,20),(25,25),(25,0),(25,25),(25,25),(25,25),(25,25),(25,25),(0,25),(20,20)]

    def timer_callback(self):
        if self.index >= len(self.actions):
            self.get_logger().info('Finished executing all actions.')
            # Stop the robot at the end
            self.publisher_.publish(Twist())  # Zero twist
            self.destroy_node()
            return
        ul, ur = self.actions[self.index]
        self.index += 1

        v_l = (RADIUS*2*np.pi*ul)/60
        v_r = (RADIUS*2*np.pi*ur)/60
      
        Phi = ((v_r - v_l)/LENGTH)

        v_straight = (v_l + v_r)/2
        msg = Twist()
        
        if v_l == v_r:
            print("Going straight")
            msg.linear.x = v_straight
            msg.angular.z = 0.0
        if Phi<0 :
            print("Turning right")
            msg.linear.x = v_straight
            msg.angular.z=Phi
        elif Phi>0 :
            print("Turning left")
            msg.linear.x = v_straight
            msg.angular.z=Phi

        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = PathExecutor()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
