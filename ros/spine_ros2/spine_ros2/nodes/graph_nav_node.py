#!/usr/bin/env python
import enum
from typing import List, Optional, Tuple, Union
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
import uuid
import tf2_ros
from tf2_ros import TransformException
import threading

from action_msgs.msg import GoalStatusArray, GoalStatus  # ROS2 action status array
from nav2_msgs.action import NavigateToPose  # ROS2 Nav2 actions
from std_msgs.msg import Header
from scipy.spatial.transform import Rotation

from geometry_msgs.msg import Pose, PoseStamped
from spine.mapping.graph_util import GraphHandler
from spine_interface_ros2.srv import (
    AddNode,
    Task,

)


def unit_vector(vector):
    return vector / np.linalg.norm(vector)


class NAV_STATUS(enum.Enum):
    # NONE = 0
    # GOAL_IN_PROGRESS = 1
    # GOAL_CANCELED = 2
    # GOAL_COMPLETE = 3
    # FAILED_TO_FIND_PLAN = 4
    # REJECTED = 5
    # PREMPTING = 6
    # RECALLING = 7
    # RECALLED = 8
    # LOST = 9

    STATUS_UNKNOWN=0
    STATUS_ACCEPTED=1
    STATUS_EXECUTING=2
    STATUS_CANCELING=3
    STATUS_SUCCEEDED=4
    STATUS_CANCELED=5
    STATUS_ABORTED=6

class GraphNavNode(Node):
    ZERO_VEC_2D = np.zeros(
        2,
    )
    DEFAULT_GRAPH = "src/planning_ros_pkgs/SPINE/ros/spine_ros2/data/graph.json"
    def __init__(self) -> None:
        super().__init__("graph_nav_node")

        self.declare_parameter("ns", "/")
        self.ns = self.get_parameter("ns").value
        self.declare_parameter("nav2_action", "navigate_to_pose")  # Nav2 action name

        self.declare_parameter("robot_frame", "/husky/base_link")
        self.declare_parameter("world_frame", "/world")
        self.declare_parameter("max_goal_dist_m", 9.0)

        self.declare_parameter("goal_reached_lin_tol", 0.5)

        # self.declare_parameter("pub", f"/{self.ns}/move_base/goal") # Not used in ros2
        # self.declare_parameter("cancel", f"/{self.ns}/move_base/cancel") # not used in ros2
        self.declare_parameter("graph","")  # , self.DEFAULT_GRAPH)

        self.ns = self.get_parameter("ns").value
        self.robot_frame = self.get_parameter("robot_frame").value
        self.world_frame = self.get_parameter("world_frame").value
        self.max_goal_dist_m = float(self.get_parameter("max_goal_dist_m").value)
        self.goal_reached_lin_tol = float(self.get_parameter("goal_reached_lin_tol").value)

        # # Keep these variables, even though ROS2 version uses an ActionClient instead of publishers
        # pub_topic = self.get_parameter("pub").value
        # cancel_topic = self.get_parameter("cancel").value
        graph = self.get_parameter("graph").value
        # for navigation
        self.declare_parameter("object_goal_angle_deg", 30)
        object_goal_angle_tol = self.get_parameter("object_goal_angle_deg").value
        self.object_goal_angle_tol = np.deg2rad(object_goal_angle_tol)

        # for timeouts
        self.declare_parameter("timeout_s", 20)
        self.declare_parameter("timeout_dist_m", 0.25)
        self.timeout_s = float(self.get_parameter("timeout_s").value)
        self.timeout_dist_m = float(self.get_parameter("timeout_dist_m").value)

        self.current_goal_id = 0

        # TF2 (ROS2) 
        self.tf_buffer = tf2_ros.Buffer()
        # self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self,spin_thread=True)
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.cb_group = ReentrantCallbackGroup()

        ## This is listening to /tf but we are publishing to /a200_0000/tf
        self.graph = GraphHandler(graph)
        # self.current_nav_status = NAV_STATUS.NONE

        # Nav2 Action client (replaces move_base goal/cancel publishers) ----
        # self.declare_parameter("nav2_action", "navigate_to_pose")  # Nav2 action name
        action_name = self.get_parameter("nav2_action").value
        if self.ns and self.ns != "/":
            # action topic commonly is namespaced like "/<ns>/navigate_to_pose"
            action_name = f"/{self.ns.strip('/')}/{action_name.strip('/')}"
        else:
            action_name = f"/{action_name.strip('/')}"

        self.action_name = action_name
        self.nav_action_client = ActionClient(self, NavigateToPose, self.action_name,callback_group=self.cb_group)
        self.current_goal_handle = None
        self.current_goal_uuid = None

        # They are not used in ROS2 Nav2 action version
        # self.pub = None
        # self.cancel_goal_pub = None

        region_goal = f"/{self.ns}/region_goal"
        object_goal = f"/{self.ns}/object_goal"
        add_node = f"/{self.ns}/add_node"
        
        ### Services
        self.region_sub = self.create_service(Task, region_goal, self.region_goal_cbk, callback_group=self.cb_group)
        self.object_sub = self.create_service(Task, object_goal, self.object_goal_cbk, callback_group=self.cb_group)
        self.add_node_sub = self.create_service(AddNode, add_node, self.add_node_cbk, callback_group=self.cb_group)
        
        ## Publisher
        # self.pub = rospy.Publisher(pub_topic, MoveBaseActionGoal)
        # self.cancel_goal_pub = rospy.Publisher(cancel_topic, GoalID)

        # Subscriber
        status_topic = f"{self.action_name}/_action/status"
        # self.get_logger().info(f" status callback is {status_topic}")

        self.nav_status_sub = self.create_subscription(
            GoalStatusArray, status_topic, self.nav_status_cbk, 10, callback_group=self.cb_group
        )


    def add_node_cbk(self, req: AddNode.Request, resp: AddNode.Response) -> AddNode.Response:
        # TODO should this be flipped
        attrs = {"coords": [req.x, req.y], "type": req.type}
        self.graph.update_with_node(node=req.node_id, attrs=attrs, edges=req.neighbors)
        self.get_logger().debug("updating graph with coords")
        resp = AddNode.Response()
        resp.success = True
        return resp

    def nav_status_cbk(self, status: GoalStatusArray) -> None:
        # self.get_logger().info(f"HI {str(status.status_list[-1].status)}")
        if len(status.status_list):
            self.current_nav_status = NAV_STATUS(status.status_list[-1].status)
            self.get_logger().info(f" Inside nav_status_cbk the nav status result is {str(self.current_nav_status)}")


    def lookup_robot_pose(self) -> Tuple[Tuple[List[int], List[int]], bool]:
        timeout = Duration(seconds=1)
        try:
            req_time = Time()

            # Wait a bit for TF to be available (prevents startup/sim-time gaps)
            if not self.tf_buffer.can_transform(self.world_frame, self.robot_frame, req_time, timeout=timeout):
                self.get_logger().warn(f"TF not available yet: {self.world_frame} -> {self.robot_frame}")
                return (None, None), False

            # self.get_logger().info(("{}, {}").format(self.world_frame, self.robot_frame))
            # ROS2 TF2 lookup gives TransformStamped
            t = self.tf_buffer.lookup_transform(
                self.world_frame,
                self.robot_frame,
                rclpy.time.Time()
            )
            pos = (t.transform.translation.x, t.transform.translation.y, t.transform.translation.z)
            quat = (
                t.transform.rotation.x,
                t.transform.rotation.y,
                t.transform.rotation.z,
                t.transform.rotation.w,
            )
            return (pos, quat), True

        except tf2_ros.ExtrapolationException as e:
            # If time mismatch happens, retry "latest" explicitly
            self.get_logger().warn(f"TF extrapolation for {self.world_frame}->{self.robot_frame}: {e}. Retrying latest.")
            try:
                if not self.tf_buffer.can_transform(self.world_frame, self.robot_frame, Time(), timeout=timeout):
                    return (None, None), False
                t = self.tf_buffer.lookup_transform(self.world_frame, self.robot_frame, Time(), timeout=timeout)

                pos = (t.transform.translation.x, t.transform.translation.y, t.transform.translation.z)
                quat = (
                    t.transform.rotation.x,
                    t.transform.rotation.y,
                    t.transform.rotation.z,
                    t.transform.rotation.w,
                )
                return (pos, quat), True
            except Exception as e2:
                self.get_logger().warn(f"TF retry failed for {self.world_frameld}->{self.robot_frame}: {e2}")
                return (None, None), False

        except TransformException as e:
            self.get_logger().warn(f"TF lookup failed for {self.world_frame}->{self.robot_frame}: {e}")
            return (None, None), False



    def get_goal_angle_yaw(
        self, goal_point: np.ndarray, obj_point: Union[np.ndarray, None] = None
    ) -> float:
        goal_angle = np.arctan2(goal_point[1], goal_point[0])
        if obj_point is not None:
            goal_to_obj = obj_point - goal_point
            obj_angle = np.arctan2(goal_to_obj[1], goal_to_obj[0])
            goal_angle = obj_angle

        return goal_angle

    def dist_from_goal(self) -> bool:
        pass

    def wait_for_nav_success(self) -> bool:
        # TODO bypass for real experiments
        while False and self.current_nav_status != NAV_STATUS.STATUS_SUCCEEDED:
            self.get_logger().info(f"waiting for nav success. status: {self.current_nav_status}")
            time.sleep(5)
        return True

    def _normalized_angle_diff(self, angle_1: float, angle_2: float) -> float:
        diff = angle_1 - angle_2
        diff = (diff + np.pi) % (2 * np.pi) - np.pi

        return np.abs(diff)

    def get_min_dist_to_buffer(self, pos: np.ndarray, buffer: List[np.array]) -> float:
        # nothing to compare
        if len(buffer) == 0:
            return np.inf
        return np.linalg.norm(pos - np.array(buffer).reshape(-1, 2), axis=-1).min()

    def wait_for_goal_reached(
        self, goal_point, goal_angle=None, tol=5, angle_tol=0.5
    ) -> bool:
        
        self.get_logger().info("wait for goal")
        # self.get_logger().info(self.current_nav_status)

        in_lin_tol = lambda pos: np.linalg.norm(pos - goal_point, ord=2) < tol
        in_angle_tol = (
            lambda angle: self._normalized_angle_diff(angle_1=angle, angle_2=goal_angle)
            < angle_tol
        )

        if goal_angle != None:
            stop_condition = lambda pos, yaw: in_lin_tol(pos) and in_angle_tol(yaw)
        else:
            stop_condition = lambda pos, yaw: in_lin_tol(pos)

        start_time = self.get_clock().now()
        history_buffer = []
        pos, _ = self.get_robot_position()
        history_buffer.append(pos)

        while True:
            pos, yaw = self.get_robot_position()
            # TODO debugging
            if True or goal_angle != None:
                dist = np.linalg.norm(pos - goal_point, ord=2)
                # rospy.loginfo(f"dist: {dist}, tol: {tol}")
                # rospy.loginfo(
                #     f"current pose ({pos}, {yaw}), desired: ({goal_point} ,{goal_angle})"
                #     f", tols: ({tol}, {angle_tol})"
                # )
            if stop_condition(pos, yaw):
                break

            # break if can't reach goal
            if (
                self.current_nav_status == NAV_STATUS.STATUS_ABORTED
                or self.current_nav_status == NAV_STATUS.STATUS_UNKNOWN
            ):
                self.cancel_goal()
                return False

            # if robot is moving, reset timer and update history
            if self.get_min_dist_to_buffer(pos, history_buffer) > self.timeout_dist_m:
                start_time = self.get_clock().now()
                history_buffer.append(pos)

            # if robot has been stationary for a while, consider goal failed.

            elapsed = (self.get_clock().now() - start_time).nanoseconds / 1e9
            if elapsed > self.timeout_s:
                self.get_logger().info(f"Goal taking over timeout ({self.timeout_s}). Cancelling")

                self.cancel_goal()
                return False

            time.sleep(0.5)

        # wait for goal to be cancelled
        self.cancel_goal()
        return True

    def command_subgoals(
        self, robot_position: np.ndarray, goal_point: np.ndarray
    ) -> bool:
        diff = goal_point - robot_position
        dist = np.linalg.norm(diff, ord=2)

        unit_vec = diff / dist
        unit_angle = self.get_goal_angle_yaw(goal_point=unit_vec)

        n_segments = int(dist // self.max_goal_dist_m) 
        segments = [
            robot_position + unit_vec * self.max_goal_dist_m * (i + 1)
            for i in range(n_segments)
        ]

        for segment in segments:
            self.pub_msg(goal_point=segment, orientation_yaw=unit_angle)
            time.sleep(5)  # debounce

            # don't check angle TODO check this
            success = self.wait_for_goal_reached(
                goal_point=segment,
                goal_angle=None,
                tol=self.goal_reached_lin_tol,
                angle_tol=4,
            )
            if not success:
                return False
            # self.wait_for_nav_success()

        return True

    def get_robot_position(self) -> Tuple[np.ndarray, float]:
        robot_pose, transform_found = self.lookup_robot_pose()

        # if not transform_found:
        #     raise ValueError("No transform found. Couldn't plan")
        
        if not transform_found:
            # don't crash the node; just fail this planning cycle
            raise RuntimeError("TF not ready (map->base_link). Cannot plan yet.")

        robot_position = np.array(robot_pose[0])[:2]  # only care about xy
        yaw = Rotation.from_quat(robot_pose[1]).as_euler("xyz")[2]
        return robot_position, yaw

    def intermediate_nav(self, goal_point: np.ndarray) -> Union[bool, str]:
        """Move base cannot command goals far (as defined by a threshold) away
        from the robot. This breaks down long goals into intermediate subgoals
        which are realized, until `goal_point` is reachable by one command.

        Parameters
        ----------
        goal_point : np.ndarray

        Returns
        -------
        Union[bool, str]
        """
        robot_position, _ = self.get_robot_position()
        dist = np.linalg.norm(goal_point - robot_position, ord=2)
        # self.get_logger().info(goal_point)
        # goal is to far away. navigate by subgoals
        if dist > self.max_goal_dist_m:
            success = self.command_subgoals(
                robot_position=robot_position, goal_point=goal_point
            )
            if not success:
                return False, "Failed to reach goal"

        return True, ""

    def object_goal_cbk(self, goal_task: Task.Request, resp: Task.Response) -> Task.Response:
        goal_node = goal_task.task
        (obj, obj_attr), (region, region_attr), found = self.graph.lookup_object(
            goal_node
        )

        if not found:
            resp.success = False
            resp.message = f"Could not find node: {goal_node}"
            return resp


        goal_point = np.array(region_attr["coords"])
        goal_angle = self.get_goal_angle_yaw(
            goal_point=goal_point, obj_point=np.array(obj_attr["coords"])
        )

        # if goal is too far to directly navigate to
        # success, msg = self.intermediate_nav(goal_point=goal_point)
        # if not success:
        #     resp.success = success
        #     resp.message = msg
        #     return resp

        self.pub_msg(goal_point=goal_point, orientation_yaw=goal_angle)
        success = self.wait_for_nav_success()
        success = self.wait_for_goal_reached(
            goal_point=goal_point,
            goal_angle=goal_angle,
            tol=self.goal_reached_lin_tol,
            angle_tol=self.object_goal_angle_tol,
        )
        resp.success = bool(success)
        resp.message = "reached goal" if success else "failed"
        return resp

    def region_goal_cbk(self, goal: Task.Request, resp: Task.Response) -> Task.Response:
        self.get_logger().info(" In the regional goal callback")
        goal_node = goal.task
        attr, found = self.graph.lookup_node(goal_node)

        if not found:
            resp.success = False
            resp.message = f"could not find goal: {goal_node}"
            return resp

        goal_point = np.array(attr["coords"])

        # success, msg = self.intermediate_nav(goal_point=goal_point)
        # if not success:
        #     self.logger().info("not success after intermediate_nav")
        #     resp.success = success
        #     resp.message = msg
        #     return resp
      
        success = self.pub_msg(goal_point=goal_point)
        # success = self.wait_for_nav_success()
        # success = self.wait_for_goal_reached(
        #     goal_point=goal_point, tol=self.goal_reached_lin_tol
        # )
        
        resp.success = success
        resp.message = "reached goal" if success else "failed"
        self.get_logger().info("At the end of region_goal_cbk {}".format(success))
        return resp


    def form_msg(self, goal: np.ndarray, orientation_yaw: float = 0) -> PoseStamped:
        assert goal.ndim == 1
        if len(goal) == 2:
            goal = np.pad(goal, (0, 1))

        self.current_goal_id += 1

        quat = Rotation.from_euler("xyz", (0, 0, orientation_yaw)).as_quat()

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.world_frame
        # self.get_logger().info(self.world_frame)
        pose = Pose()
        pose.orientation.x = float(quat[0])
        pose.orientation.y = float(quat[1])
        pose.orientation.z = float(quat[2])
        pose.orientation.w = float(quat[3])

        pose.position.x = float(goal[0])
        pose.position.y = float(goal[1])
        pose.position.z = float(goal[2])

        msg = PoseStamped(header=header, pose=pose)

        # goal_msg = MoveBaseActionGoal()
        # goal_msg.header = msg.header
        # goal_msg.goal_id = GoalID(stamp=msg.header.stamp, id=str(self.current_goal_id))
        # goal_msg.goal.target_pose = msg
        # return goal_msg
        # self.get_logger().info("form_msg")
        # self.get_logger().info(pose.position.x)
        return msg



    def pub_msg(self, goal_point, orientation_yaw=0.0) -> bool:
        msg = self.form_msg(goal_point, orientation_yaw=orientation_yaw)

        if not self.nav_action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(f"Nav2 action server not available: {self.action_name}")
            return False

        goal = NavigateToPose.Goal()
        goal.pose = msg
        self.get_logger().info(f"The goal is {goal.pose}")

        done_event = threading.Event()
        nav_result = {"success": False}

        def on_result(result_future):
            try:
                result_wrap = result_future.result()
                status = result_wrap.status
                self.get_logger().info(f"Nav result status={status}, result={result_wrap.result}")
                nav_result["success"] = (status == GoalStatus.STATUS_SUCCEEDED)
            except Exception as e:
                self.get_logger().error(f"Failed getting nav result: {e}")
                nav_result["success"] = False
            finally:
                done_event.set()

        def on_goal_response(send_future):
            try:
                goal_handle = send_future.result()
                if goal_handle is None or not goal_handle.accepted:
                    self.get_logger().warn("Goal rejected")
                    nav_result["success"] = False
                    done_event.set()
                    return

                self.get_logger().info("Goal accepted")
                result_future = goal_handle.get_result_async()
                result_future.add_done_callback(on_result)

            except Exception as e:
                self.get_logger().error(f"Goal response failed: {e}")
                nav_result["success"] = False
                done_event.set()

        send_future = self.nav_action_client.send_goal_async(goal)
        send_future.add_done_callback(on_goal_response)

        if not done_event.wait(timeout=120.0):
            self.get_logger().error("Timed out waiting for nav result")
            return False

        return nav_result["success"]
    
    # def pub_msg(
    #     self, goal_point: np.ndarray, orientation_yaw: Optional[float] = 0
    # ) -> None:
    #     # self.update_controller()
    #     # self.get_logger().info("pub_msg")
    #     # self.get_logger().info(goal_point)

    #     msg = self.form_msg(goal_point, orientation_yaw=orientation_yaw)

    #     # self.pub.publish(msg) # not used in ros2
    #     # Wait for Nav2 action server
    #     if not self.nav_action_client.wait_for_server(timeout_sec=5.0):
    #         self.get_logger().error(f"Nav2 action server not available: {self.action_name}")
    #         return

    #     goal = NavigateToPose.Goal()
    #     goal.pose = msg
    #     # self.get_logger().info("in pub_msg")
    #     self.get_logger().info("The gaol is {}".format(goal.pose))
    #     # Track goal id locally
    #     self.current_goal_uuid = uuid.uuid4()

    #     # Send goal
    #     send_future = self.nav_action_client.send_goal_async(goal)
    #     self.get_logger().info("after send_future")

    #     rclpy.spin_until_future_complete(self, send_future)
    #     goal_handle = send_future.result()

    #     if goal_handle is None:
    #         self.get_logger().error("Goal response future returned None")
    #         return False

    #     if not goal_handle.accepted:
    #         self.get_logger().warn("Goal was rejected by server")
    #         return False

    #     self.get_logger().info("Goal accepted")

    #     result_future = goal_handle.get_result_async()
    #     rclpy.spin_until_future_complete(self, result_future)

    #     result_wrap = result_future.result()
    #     if result_wrap is None:
    #         self.get_logger().error("Navigation result future returned None")
    #         return False

    #     status = result_wrap.status
    #     result = result_wrap.result
    #     self.get_logger().info(f"Nav result status={status}, result={result}")

    #     return status == GoalStatus.STATUS_SUCCEEDED


    def _on_goal_response(self, future):
        goal_handle = future.result()
        if goal_handle is None:
            self.get_logger().error("Goal response future returned None")
            return

        if not goal_handle.accepted:
            self.get_logger().warn("Goal was rejected by server")
            return

        self.get_logger().info("Goal accepted")

        # Now wait for result async (non-blocking)
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_nav_result)

    def _on_nav_result(self, future):
        try:
            result = future.result().result   # nav2_msgs/action/NavigateToPose.Result
            status = future.result().status   # GoalStatus code
            self.get_logger().info(f"Nav result status={status}, result={result}")
        except Exception as e:
            self.get_logger().error(f"Failed getting nav result: {e}")


    def cancel_goal(self) -> None:
        # while self.current_nav_status == NAV_STATUS.STATUS_EXECUTING:
        # for _ in range(5):  # there is a delay in reading status, so use time for now
        #     self.cancel_goal_pub.publish(
        #         GoalID(stamp=rospy.Time.now(), id=str(self.current_goal_id))
        #     )
        #     rospy.sleep(0.1)
        if self.current_goal_handle is None:
            return

        cancel_future = self.current_goal_handle.cancel_goal_async()
        rclpy.spin_until_future_complete(self, cancel_future)
        self.current_nav_status = NAV_STATUS.STATUS_CANCELED
        if cancel_future.result() is not None:
            self.get_logger().info("Cancel request accepted")
        self.get_logger().info("cancel_goal:")
        self.get_logger().info(f"The goal should be canceled and result={self.current_nav_status}")


def main():
    rclpy.init()
    nav = GraphNavNode()
    # rclpy.spin(nav)
    # nav.destroy_node()
    # rclpy.shutdown()
    executor = MultiThreadedExecutor()
    executor.add_node(nav)
    executor.spin()
    nav.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
