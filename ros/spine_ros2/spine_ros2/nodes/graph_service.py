#!/usr/bin/env python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from spine.mapping.graph_util import GraphHandler
import json

from spine_interface_ros2.srv import Graph


class GraphServiceNode(Node):
    def __init__(self) -> None:
        super().__init__("graph_service")
        self.get_logger().info("garph service node started")

        self.publish_graph_srv = self.create_service(Graph, "graph_srv", self.graph_srv)

        self.spine_graph_srv = self.create_publisher(String, "/titan/overhead_graph", 10)

    def graph_srv(self, graph_path: Graph.Request, resp:  Graph.Response) :
        """Takes a path to JSON containing scene graph."""
        path = graph_path.graph
        current_location = graph_path.current_location
        self.get_logger().info(f"trying to load graph")

        origin_data = {}
        # with open(path) as f:
        #     data = json.load(f)

        #     if "origin" in data:
        #         origin_data["origin"] = data["origin"]

        try:
            self.get_logger().info(f"trying to load: {path}")
            graph_handler = GraphHandler(path, current_location)
            graph_json_str = graph_handler.to_json_str(extra_data=origin_data)

            self.get_logger().info(f"graph as str: {graph_json_str}")

            msg = String()
            msg.data = graph_json_str
            self.spine_graph_srv.publish(msg)

            resp = Graph.Response()
            resp.success = True
            return resp
        except Exception as ex:
            self.get_logger().error(f"Graph service failed: {ex}")
            resp = Graph.Response()
            resp.success = False           
            return resp

def main(args=None):
    rclpy.init(args=args)
    node = GraphServiceNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

