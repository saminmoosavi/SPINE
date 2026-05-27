#!/usr/bin/env python3

import json
import math

import rclpy
from rclpy.node import Node

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point


class FactoryWorldVisualizer(Node):

    def __init__(self):
        super().__init__("factory_world_visualizer")

        self.json_file = self.declare_parameter(
            "json_file",
            "src/planning_ros_pkgs/SPINE/ros/spine_ros2/data/graph.json"
        ).value

        self.frame_id = self.declare_parameter(
            "frame_id",
            "map"
        ).value

        self.marker_topic = self.declare_parameter(
            "marker_topic",
            "/factory_world_markers"
        ).value

        self.pub = self.create_publisher(
            MarkerArray,
            self.marker_topic,
            10
        )

        self.world = self.load_world(self.json_file)

        self.timer = self.create_timer(
            1.0,
            self.publish_markers
        )

        self.get_logger().info(
            f"Publishing factory world markers from {self.json_file}"
        )

    def load_world(self, path):
        with open(path, "r") as f:
            return json.load(f)

    def publish_markers(self):
        marker_array = MarkerArray()
        marker_id = 0

        stamp = self.get_clock().now().to_msg()

        # Clear old markers
        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)

        # ----------------------------------------------------
        # Publish regions
        # ----------------------------------------------------
        region_lookup = {}

        for region in self.world.get("regions", []):
            name = region["name"]
            x, y = region["coords"]
            region_lookup[name] = (x, y)

            marker_array.markers.append(
                self.make_sphere_marker(
                    marker_id,
                    stamp,
                    x,
                    y,
                    z=0.05,
                    scale=0.35,
                    color=(0.0, 0.4, 1.0, 1.0),
                    ns="regions"
                )
            )
            marker_id += 1

            marker_array.markers.append(
                self.make_text_marker(
                    marker_id,
                    stamp,
                    x,
                    y,
                    z=0.7,
                    text=name,
                    scale=0.5,
                    color=(0.0, 0.0, 0.0, 1.0),
                    ns="region_labels"
                )
            )
            marker_id += 1

        # ----------------------------------------------------
        # Publish region connections
        # ----------------------------------------------------
        for connection in self.world.get("region_connections", []):
            r1, r2 = connection

            if r1 not in region_lookup or r2 not in region_lookup:
                continue

            x1, y1 = region_lookup[r1]
            x2, y2 = region_lookup[r2]

            marker_array.markers.append(
                self.make_line_marker(
                    marker_id,
                    stamp,
                    x1,
                    y1,
                    x2,
                    y2,
                    z=0.02,
                    width=0.08,
                    color=(0.0, 0.8, 1.0, 0.8),
                    ns="region_connections"
                )
            )
            marker_id += 1

        # ----------------------------------------------------
        # Publish objects
        # ----------------------------------------------------
        for obj in self.world.get("objects", []):
            name = obj["name"]
            x, y = obj["coords"]

            color, scale = self.get_object_style(name)

            marker_array.markers.append(
                self.make_cube_marker(
                    marker_id,
                    stamp,
                    x,
                    y,
                    z=0.15,
                    scale=scale,
                    color=color,
                    ns="objects"
                )
            )
            marker_id += 1

            marker_array.markers.append(
                self.make_text_marker(
                    marker_id,
                    stamp,
                    x,
                    y,
                    z=0.9,
                    text=name,
                    scale=0.5,
                    color=(0.0, 0.0, 0.0, 1.0),
                    ns="object_labels"
                )
            )
            marker_id += 1

        # ----------------------------------------------------
        # Publish robot location
        # ----------------------------------------------------
        robot_location = self.world.get("robot_location", None)

        if robot_location and robot_location in region_lookup:
            x, y = region_lookup[robot_location]

            marker_array.markers.append(
                self.make_sphere_marker(
                    marker_id,
                    stamp,
                    x,
                    y,
                    z=0.2,
                    scale=0.7,
                    color=(0.0, 1.0, 0.0, 1.0),
                    ns="robot_location"
                )
            )
            marker_id += 1

            marker_array.markers.append(
                self.make_text_marker(
                    marker_id,
                    stamp,
                    x,
                    y,
                    z=1.1,
                    text=f"ROBOT: {robot_location}",
                    scale=0.4,
                    color=(0.0, 0.0, 0.0, 1.0),
                    ns="robot_label"
                )
            )
            marker_id += 1

        self.pub.publish(marker_array)

    # --------------------------------------------------------
    # Marker helpers
    # --------------------------------------------------------

    def make_sphere_marker(self, marker_id, stamp, x, y, z, scale, color, ns):
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = stamp
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        marker.pose.position.z = float(z)
        marker.pose.orientation.w = 1.0

        marker.scale.x = float(scale)
        marker.scale.y = float(scale)
        marker.scale.z = float(scale)

        marker.color.r = float(color[0])
        marker.color.g = float(color[1])
        marker.color.b = float(color[2])
        marker.color.a = float(color[3])

        return marker

    def make_cube_marker(self, marker_id, stamp, x, y, z, scale, color, ns):
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = stamp
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        marker.pose.position.z = float(z)
        marker.pose.orientation.w = 1.0

        marker.scale.x = float(scale[0])
        marker.scale.y = float(scale[1])
        marker.scale.z = float(scale[2])

        marker.color.r = float(color[0])
        marker.color.g = float(color[1])
        marker.color.b = float(color[2])
        marker.color.a = float(color[3])

        return marker

    def make_text_marker(self, marker_id, stamp, x, y, z, text, scale, color, ns):
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = stamp
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD

        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        marker.pose.position.z = float(z)
        marker.pose.orientation.w = 1.0

        marker.scale.z = float(scale)

        marker.color.r = float(color[0])
        marker.color.g = float(color[1])
        marker.color.b = float(color[2])
        marker.color.a = float(color[3])

        marker.text = text

        return marker

    def make_line_marker(self, marker_id, stamp, x1, y1, x2, y2, z, width, color, ns):
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = stamp
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD

        marker.pose.orientation.w = 1.0

        marker.scale.x = float(width)

        marker.color.r = float(color[0])
        marker.color.g = float(color[1])
        marker.color.b = float(color[2])
        marker.color.a = float(color[3])

        p1 = Point()
        p1.x = float(x1)
        p1.y = float(y1)
        p1.z = float(z)

        p2 = Point()
        p2.x = float(x2)
        p2.y = float(y2)
        p2.z = float(z)

        marker.points.append(p1)
        marker.points.append(p2)

        return marker

    def get_object_style(self, name):
        if name.startswith("person"):
            return (1.0, 0.0, 0.0, 1.0), (0.5, 0.5, 1.7)

        if name.startswith("column"):
            return (0.5, 0.5, 0.5, 1.0), (0.6, 0.6, 2.5)

        if name.startswith("short_shelf"):
            return (0.7, 0.4, 0.1, 1.0), (1.2, 0.5, 0.8)

        if name.startswith("tall_shelf"):
            return (0.4, 0.2, 0.1, 1.0), (1.4, 0.6, 2.0)

        if name.startswith("chair"):
            return (0.7, 0.0, 0.7, 1.0), (0.5, 0.5, 0.7)

        if name.startswith("table"):
            return (0.0, 0.7, 0.7, 1.0), (1.0, 0.8, 0.7)

        return (1.0, 1.0, 1.0, 1.0), (0.5, 0.5, 0.5)


def main(args=None):
    rclpy.init(args=args)
    node = FactoryWorldVisualizer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()