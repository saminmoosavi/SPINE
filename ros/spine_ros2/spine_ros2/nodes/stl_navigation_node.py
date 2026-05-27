#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from geometry_msgs.msg import PoseStamped

from nav2_msgs.action import FollowWaypoints


class WaypointNavigation(Node):

    def __init__(self):

        # Default robot namespace
        super().__init__('follow_path_client')

        self.ns = self.declare_parameter("ns", "/a200_0000").value

        # Namespaced FollowPath action
        self.client = ActionClient(
            self,
            FollowWaypoints,
            f'{self.ns}/follow_waypoints'
        )
        
        self.get_logger().info(
            f"Waiting for action server: "
            f"{self.ns}/follow_waypoints"
        )

        self.client.wait_for_server()

        self.send_goal()

    def create_pose(self, x, y, yaw=0.0):

        pose = PoseStamped()

        pose.header.frame_id = "map"
        pose.header.stamp = self.get_clock().now().to_msg()

        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.0

        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)

        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw

        return pose

    def build_trajectory(self, waypoints):

        poses = []

        for i, (x, y) in enumerate(waypoints):

            if i < len(waypoints) - 1:
                nx, ny = waypoints[i + 1]
                yaw = math.atan2(ny - y, nx - x)
            else:
                yaw = 0.0

            poses.append(
                self.create_pose(x, y, yaw)
            )


        return poses

    def send_goal(self):

        goal_msg = FollowWaypoints.Goal()
        # Example waypoints
        waypoints = [
            (-13.0, 14.00),
            (-12.5, 20.0),
            (-9.0, 23.6),
        ]
        goal_msg.poses = self.build_trajectory(waypoints)


        self.get_logger().info("Sending waypoint goal")
        # navigator.followWaypoints(goal_poses)
 
        future = self.client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )

        future.add_done_callback(
            self.goal_response_callback
        )

    def goal_response_callback(self, future):

        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().error("Goal rejected")
            return

        self.get_logger().info("Goal accepted")

        result_future = goal_handle.get_result_async()

        result_future.add_done_callback(
            self.result_callback
        )

    def feedback_callback(self, feedback_msg):

        feedback = feedback_msg.feedback

        self.get_logger().info(
            f"Current waypoint index: {feedback.current_waypoint}"
        )
    def result_callback(self, future):

        result = future.result().result

        if result.missed_waypoints:
            self.get_logger().warn(
                f"Missed waypoints: {result.missed_waypoints}"
            )
        else:
            self.get_logger().info("All waypoints completed successfully")

        # rclpy.shutdown()


def main(args=None):

    rclpy.init(args=args)

    node = WaypointNavigation()

    rclpy.spin(node)

    rclpy.shutdown()


if __name__ == '__main__':
    main()