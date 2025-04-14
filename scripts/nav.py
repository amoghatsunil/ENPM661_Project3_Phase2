#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math
from pathlib import Path
import csv 
import time

class GoToPoint(Node):
    def __init__(self):
        super().__init__('go_to_point')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        # self.timer = self.create_timer(0.1, self.control_loop)
        self.goal_list = self.read_goals_from_csv('path.csv')
        self.goal_list = self.filter_path_by_distance(self.goal_list,0.2)
        self.go_to_waypoints()

    def go_to_waypoints(self):
        current_x, current_y = -0.5, 2.5
        current_theta = -math.pi # Assume facing "positive x" at the start

        for goal_x, goal_y in self.goal_list:
            # 1. Compute angle to goal
            delta_x = goal_x - current_x
            delta_y = goal_y - current_y
            target_angle = math.atan2(delta_y, delta_x)

            # 2. Compute minimal angle to rotate
            angle_to_rotate = target_angle - current_theta
            angle_to_rotate = math.atan2(math.sin(angle_to_rotate), math.cos(angle_to_rotate))  # Normalize to [-π, π]

            # 3. Rotate
            print("angle to rotate:  ",angle_to_rotate)
            self.rotate(angle_to_rotate)

            # 4. Compute distance to goal
            distance = math.hypot(delta_x, delta_y)
            print("distance : ",distance)
            
            # 5. Move forward
            self.move_straight(distance)

            # 6. Update internal state
            current_x = goal_x
            current_y = goal_y
            current_theta = target_angle

        self.stop_robot()

    def rotate(self, angle, angular_speed=0.1):
        twist = Twist()
        twist.angular.z = angular_speed if angle > 0 else -angular_speed

        rotated = 0.0
        dt = 0.02
        rate = self.create_rate(1/dt)

        while abs(rotated) < abs(angle):
            self.publisher_.publish(twist)
            rotated += angular_speed * dt
            print(f"Rotated: {rotated:.2f} / {angle:.2f}")
            time.sleep(dt)

        self.stop_robot()

    def move_straight(self, distance, linear_speed=0.1):
        twist = Twist()
        twist.linear.x = linear_speed

        moved = 0.0
        dt = 0.04
        rate = self.create_rate(1/dt)

        while moved < distance:
            self.publisher_.publish(twist)
            moved += linear_speed * dt
            print(f"Moved: {moved:.2f} / {distance:.2f}")
            time.sleep(dt)

        self.stop_robot()

    def stop_robot(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.publisher_.publish(twist)

    def filter_path_by_distance(self, path, d_thresh):
        if not path:
            return []

        filtered = [path[0]]
        for point in path[1:-1]:  
            prev = filtered[-1]
            dist = math.hypot(point[0] - prev[0], point[1] - prev[1])
            if dist >= d_thresh:
                filtered.append(point)

        filtered.append(path[-1])

        # Coordinate transformation
        filtered_adjusted = []
        for point in filtered:
            new_x = -point[0]
            new_y = 3.0 - point[1]
            filtered_adjusted.append((new_x, new_y))

        return filtered_adjusted

    def read_goals_from_csv(self, filename):
        path = Path(__file__).parent / filename
        goals = []
        with open(path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                x = float(row['x'])
                y = float(row['y'])
                goals.append((x, y))
        return goals

def main(args=None):
    rclpy.init(args=args)
    node = GoToPoint()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()