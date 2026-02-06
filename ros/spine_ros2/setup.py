from setuptools import setup
from glob import glob
import os
package_name = "spine_ros2"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/spine.launch.py"]),
        (os.path.join("share", package_name, "data"), glob("data/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Samin Moosavi",
    maintainer_email="saminmoosavi@yahoo.com",
    description="ROS 2 wrapper for SPINE semantic planner",
    license="BSD-3-Clause",
    entry_points={
        'console_scripts': [
            'tracker_with_yolo = spine_ros2.nodes.tracker_with_yolo:main' ,
            'graph_nav_node = spine_ros2.nodes.graph_nav_node:main',
            'spine_node = spine_ros2.nodes.spine_node:main',
            'graph_service = spine_ros2.nodes.graph_service:main',   
        ],
    },
)

