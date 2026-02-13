from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, PushRosNamespace
from ament_index_python.packages import get_package_share_directory
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution
import os


def generate_launch_description():

    # --------------------
    # Launch arguments
    # --------------------
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    ns = LaunchConfiguration("ns")

    graph = LaunchConfiguration("graph")
    full_graph = LaunchConfiguration("full_graph")
    init_location = LaunchConfiguration("init_location")

    robot_frame = LaunchConfiguration("robot_frame")
    world_frame = LaunchConfiguration("world_frame")

    use_sim_perception = LaunchConfiguration("use_sim_perception")

    viz_scale = LaunchConfiguration("viz_scale")
    min_obstacle_height = LaunchConfiguration("min_obstacle_height")
    max_goal_dist_m = LaunchConfiguration("max_goal_dist_m")

    min_new_region_dist = LaunchConfiguration("min_new_region_dist")
    costmap_filter_thresh = LaunchConfiguration("costmap_filter_thresh")
    costmap_filter_n_cells = LaunchConfiguration("costmap_filter_n_cells")

    should_classify_scene = LaunchConfiguration("should_classify_scene")
    label_srv = LaunchConfiguration("label_srv")
    use_open_vocab_detection = LaunchConfiguration("use_open_vocab_detection")
    scene_description = LaunchConfiguration("scene_description")

    goal_reached_lin_tol = LaunchConfiguration("goal_reached_lin_tol")
    goal_timeout = LaunchConfiguration("goal_timeout")

    spine_ros_share = get_package_share_directory("spine_ros2")

    # --------------------
    # Declare arguments
    # --------------------
    declare_args = [

        DeclareLaunchArgument("ns", default_value="ns"),
        DeclareLaunchArgument(
            "graph",
            default_value=PathJoinSubstitution([
                FindPackageShare("spine_ros2"),
                "data",
                "graph.json"
            ]),
        ),
        DeclareLaunchArgument(
            "full_graph",
            default_value=PathJoinSubstitution([
                FindPackageShare("spine_ros2"),
                "data",
                "graph.json"
            ]),
        ),

        DeclareLaunchArgument("init_location", default_value="R1"),

        DeclareLaunchArgument("robot_frame", default_value="base_link"),
        DeclareLaunchArgument("world_frame", default_value="map"),

        DeclareLaunchArgument("use_sim_perception", default_value="false"),

        DeclareLaunchArgument("viz_scale", default_value="0.5"),
        DeclareLaunchArgument("min_obstacle_height", default_value="0.4"),
        DeclareLaunchArgument("max_goal_dist_m", default_value="5.0"),

        DeclareLaunchArgument("min_new_region_dist", default_value="3"),
        DeclareLaunchArgument("costmap_filter_thresh", default_value="1"),
        DeclareLaunchArgument("costmap_filter_n_cells", default_value="3"),

        DeclareLaunchArgument("should_classify_scene", default_value="true"),
        DeclareLaunchArgument("label_srv", default_value="/grounding_dino_ros/set_labels"),
        DeclareLaunchArgument("use_open_vocab_detection", default_value="False"),
        DeclareLaunchArgument("scene_description", default_value="outdoors"),

        DeclareLaunchArgument("goal_reached_lin_tol", default_value="15.0"),
        DeclareLaunchArgument("goal_timeout", default_value="10"),
    ]
    
    # --------------------
    # garph_service.py
    # ---------------------

    graph_service = Node(
        package='spine_ros2',
        executable='graph_service',
        name='graph_service',
        output='screen',
        parameters=[{
            'ns': ns,
        }],
    )

    # --------------------
    # graph_nav_node
    # --------------------
    graph_nav_node = Node(
        package="spine_ros2",
        executable="graph_nav_node",
        name="graph_nav_node",
        output="screen",
        parameters=[
            {
                "ns": ns,
                "graph": full_graph,
                "world_frame": world_frame,
                "robot_frame": robot_frame,
                "min_obstacle_height": min_obstacle_height,
                "max_goal_dist_m": max_goal_dist_m,
                "goal_reached_lin_tol": goal_reached_lin_tol,
                "timeout_s": goal_timeout,
            }
        ],
        remappings=[
            ('/tf', 'tf'), # Remap global /tf to relative tf (becomes /my_namespace/tf)
            ('/tf_static', 'tf_static'), # Remap global /tf_static to relative tf_static
        ],
    )

    # --------------------
    # spine_node
    # --------------------
    spine_node = Node(
        package="spine_ros2",
        executable="spine_node",
        name="spine_node",
        output="screen",
        parameters=[
            {
                "ns": ns,
                "init_graph": graph,
                "full_graph": full_graph,
                "init_location": init_location,
                "sim_perception": use_sim_perception,
                "world_frame": world_frame,
                "viz_scale": viz_scale,
                "min_new_region_dist": min_new_region_dist,
                "costmap_filter_thresh": costmap_filter_thresh,
                "costmap_filter_n_cells": costmap_filter_n_cells,
                "should_classify_scene": should_classify_scene,
                "set_labels_srv": label_srv,
                "use_open_vocab_detection": use_open_vocab_detection,
                "scene_description": scene_description,
            }
        ],
    )

    # --------------------
    # Namespace group (equivalent to <group ns=...>)
    # --------------------
    group = GroupAction(
        [
            PushRosNamespace(ns),
            graph_service,
            spine_node,
            graph_nav_node,
            
        ]
    )

    return LaunchDescription(declare_args + [group])

