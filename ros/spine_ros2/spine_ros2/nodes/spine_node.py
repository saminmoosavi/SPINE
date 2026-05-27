#!/usr/bin/env python
import copy
from queue import Queue
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import json
import numpy as np
import time
import rclpy
from rclpy.node import Node
from spine.class_llm import ClassLLM
from spine.spine import SPINE
from spine.spine_util import UpdatePromptFormer
from spine.mapping.frontiers import FrontierExtractor
from spine.mapping.frontiers import Node as FrontierNode
from spine.mapping.graph_sim import GraphSim
from spine.mapping.graph_util import GraphHandler, to_float_list
from spine.viz.viz_ros import GraphViz
# from spine_ros.imu_reader import ImuReader
from spine_interface_ros2.srv import (
    AddNode,
    Task,
    Query,
    SetLabels,

)
from nav_msgs.msg import OccupancyGrid 
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Imu
from std_msgs.msg import String
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker

from spine_interface_ros2.msg import Track
from spine_ros2.utility.ros_utils import from_track_msg

# except ImportError:
#     self.get_logger().info(f"open_vocab_vision_ros is not installed. Will not read tracks")


def get_add_node_msg(
    node_id: str, x: int, y: int, type: str, neighbors: List[str]
) -> AddNode.Response:
    return AddNode.Request(
        node_id=node_id, x=int(x), y=int(y), type=type, neighbors=neighbors
    )


class SPINENode(Node):
    DEFAULT_GRAPH = (
        "src/planning_ros_pkgs/SPINE/ros/spine_ros2/data/graph.json"
    )

    def __init__(self) -> None:
        super().__init__("spine_node")
        self.ns = self.declare_parameter("ns", "").value
        graph = self.declare_parameter("init_graph").value
        full_graph = self.declare_parameter("full_graph").value
        self.current_location = self.declare_parameter("init_location", "").value
        self.use_sim_perception = self.declare_parameter("sim_perception", True).value
        self.object_track_topic = self.declare_parameter(
            "object_tracks", f"/{self.ns}/tracks"
        ).value

        self.get_logger().info(self.object_track_topic)
        # if graph is none, assume we're waiting for a graph
        # TODO tmp logic
        self.waiting_for_graph = full_graph == ""

        self.input_graph_sub = self.create_subscription(
            String, "/titan/overhead_graph",  self.input_graph_cbk, 10
        )

        # for applying transforms to graphs from other frames
        # self.imu_reader = ImuReader()

        # #
        # internal
        #
        self.action_history = []

        # #
        # for simulated perception
        # #
        if self.use_sim_perception:
            self.perception = GraphSim(
                full_graph=full_graph, init_graph=graph, init_node=self.current_location
            )
            self.graph = self.perception.working_graph
        else:
            self.graph = GraphHandler(full_graph, init_node=self.current_location)
        self.get_logger().info(str(self.graph))
        self.planner = SPINE(graph=self.graph)

        self.llm_prompt_former = UpdatePromptFormer()

        # #
        # for graph navigation
        # #
        self.navigate_to_region_srv_proxy = self.create_client(Task,
            f"/{self.ns}/graph_nav_node/region_goal" 
        )
        self.inspect_object_srv_proxy =self.create_client(Task,
            f"/{self.ns}/graph_nav_node/object_goal"
        )
        self.add_node_srv_proxy =self.create_client(AddNode,
            f"/{self.ns}/graph_nav_node/add_node"
        )
        self.task_srv = self.create_service(Task, "mission", self.task_cbk)
        self.nav_srv =self.create_service( Task, "nav_to_region",self.nav_to_region_cbk)

        self.interrupt_srv =self.create_service(Trigger, "interrupt", self.interrupt_cbk)
        self.should_interrupt = False

        self.resume_task_srv =self.create_service( Task, "resume",self.resume_task_cbk)

        # #
        # for vlm classification
        # #
        self.scene_description = self.declare_parameter("scene_description", "").value
        self.should_classify_region = self.declare_parameter("should_classify_scene", True).value
        self.classify_scene_srv_proxy =self.create_client(Trigger,
            "/vlm_infer/open_classify_scene"
        )
        self.query_scene_srv_proxy =self.create_client(Query, "/vlm_infer/query_scene")

        # #
        # other members
        # #
        self.task = ""
        self.location_history = [self.current_location]

        # #
        # for real perception
        # #
        self.tracks = {}
        self.added_tracks = set()
        self.updated_tracks = set()
        self.track_sub = self.create_subscription(
            Track, self.object_track_topic,  self.track_cbk, 10
        )
        self.track_queue = Queue()

        # #
        # for frontiers
        # #

        # #
        # for frontiers
        # #

        costmap_topic = self.declare_parameter(
            "costmap_topic", f"/{self.ns}/local_costmap/costmap"
        ).value
        self.min_new_frontier_dist = self.declare_parameter("~min_new_region_dist", 1).value
        costmap_filter_thresh = self.declare_parameter("~costmap_filter_thresh", 70).value
        costmap_filter_n_cells = self.declare_parameter("~costmap_filter_n_cells", 3).value

        self.frontier_extractor = FrontierExtractor(
            init_graph=self.graph,
            min_new_frontier_thresh=self.min_new_frontier_dist,
            costmap_filter_threshold=costmap_filter_thresh,
            costmap_filter_n_cells=costmap_filter_n_cells,
        )

        self.costmap_sub = self.create_subscription(
            OccupancyGrid, costmap_topic,  self.costmap_cbk, 10
        )
        self.filtered_costmap_pub = self.create_publisher(OccupancyGrid, "filtered_costmap", 10)

        # #
        # for setting labels
        # #
        self.class_llm = ClassLLM()
        self.use_open_vocab_detection = self.declare_parameter(
            "use_open_vocab_detection", True
        ).value
        self.space_description = self.declare_parameter("space_description", "")
        srv_topic = self.declare_parameter(
            "set_labels_srv", f"/{self.ns}/grounding_dino_ros/set_labels"
        ).value
        self.set_detection_labels_srv = self.create_client(SetLabels,srv_topic)
        self.task_label_pub = self.create_publisher(String,"/task_labels", 10)

        # #
        # for visualization
        # #
        viz_scale = self.declare_parameter("viz_scale", 0.5).value
        world_frame = self.declare_parameter("world_frame", "map").value
        self.graph_viz = GraphViz(
            graph=self.graph, scale=viz_scale, target_frame=world_frame
        )
        self.pub = self.create_publisher(Marker, "graph_viz", 100)

        # #
        # for graph logging
        # #
        self.graph_pub = self.create_publisher(String, "graph",  10)

        # for timed publishers
        self.pub_freq = 1
        self.timer = self.create_timer(
            self.pub_freq,
            self.publish_graph
        )


    def publish_graph(self):
        self.graph_viz.publish(pub=self.pub)
        self.graph_pub.publish(
            String(data=self.graph.to_json_str())
        )
        
    def input_graph_cbk(self, incoming_graph: String) -> None:
        """Receive scene graph from external provider.

        Notes
        -----
        graph may be in different coordinate system. If so, the dictionary
        containing the graph must have an `origin` key. The map is then shifted
        by the origin and east aligned.

        Parameters
        ----------
        incoming_graph : String

        Returns
        -------
        GraphResponse
        """
        filter_words = ["building"]
        # print(f"\ncurrent location: {self.current_location}")
        while not self.imu_reader.is_initialized():
            print(f"gps: {self.imu_reader.gps_initialized}")
            print(f"imu: {self.imu_reader.imu_initialized}")
            self.get_logger().info(f"waiting for imu orientation initialization")
            time.sleep(1)


        map_origin = np.array([482939.85851084325, 4421267.982947684])

        rotation_origin = self.imu_reader.get_current_rot()

        utm_origin = self.imu_reader.get_utm_origin()
        graph_as_json_str = incoming_graph.data

        data = json.loads(graph_as_json_str)

        origin = utm_origin - map_origin
        # origin = map_origin - utm_origin

        # self.get_logger().info(f"UGV origin: {utm_origin}")

        origin = np.array([0, 0])

        # if "origin" in data:
        #     graph_origin = to_float_list(data["origin"])
        #     origin = utm_origin - graph_origin
        #     self.get_logger().info(f"\n\n have origin")
        # else:
        #     origin = np.array([0, 0])

        # self.get_logger().info(f"Using origin: {origin}")

        # filter some classes out
        new_data = {
            "objects": [],
            "regions": [],
            "object_connections": [],
            "region_connections": [],
        }

        if "objects" in data.keys():
            for node in data["objects"]:
                if node["name"].split("_")[0] not in filter_words:
                    new_data["objects"].append(node)
        if "object_connections" in data:
            for connection in data["object_connections"]:
                if (
                    connection[0].split("_")[0] not in filter_words
                    and connection[1].split("_")[0] not in filter_words
                ):
                    new_data["object_connections"].append(connection)
        if "regions" in data:
            new_data["regions"] = data["regions"]
        if "region_connections" in data:
            new_data["region_connections"] = data["region_connections"]

        data = new_data

        graph_as_json_str = json.dumps(data)
        self.get_logger().info("graph dumped!")
        # TODO assume incoming graph has node road_1
        custom_data = {
            "regions": [{"name": "road_0", "coords": f"[{origin[0]}, {origin[1]}]"}],
            "region_connections": [["road_0", "road_1"]],
        }
        current_location = "road_0"

        # initialize graph, if there isn't one already
        if self.waiting_for_graph == True:
            self.waiting_for_graph = False
            success = self.graph.reset(
                graph_as_json=graph_as_json_str,
                rotation=rotation_origin,
                utm_origin=origin,
                custom_data=custom_data,
                current_location=current_location,
                flip_coords=False,
            )
            self.current_location = self.graph.current_location

            for node in self.graph.graph.nodes:
                info = self.graph.graph.nodes[node]
                neighbors = self.graph.get_neighbors(node)
                # print(node, info["coords"], info["type"], neighbors)
                response = self.add_node_srv_proxy(
                    get_add_node_msg(
                        node_id=node,
                        x=info["coords"][0],
                        y=info["coords"][1],
                        type=info["type"],
                        neighbors=[],
                    )
                )

            for node in self.graph.graph.nodes:
                info = self.graph.graph.nodes[node]
                neighbors = self.graph.get_neighbors(node)
                response = self.add_node_srv_proxy(
                    get_add_node_msg(
                        node_id=node,
                        x=info["coords"][0],
                        y=info["coords"][1],
                        type=info["type"],
                        neighbors=neighbors,
                    )
                )

            from_uav_msg = """
            Some or all of these updates are from an external mapping system running on a high-altitude UAV. These are additions to your current graph, and they are marked with the attribute: source=from_uav.
            """
            self.llm_prompt_former.update(
                freeform_updates=[from_uav_msg],
            )
            self.get_logger().info("graph existed!")


        # otherwise, treat incoming graph as regular perception update
        else:
            # print(f"processing: {graph_as_json_str}")
            tmp_handler = GraphHandler("")
            success = tmp_handler.reset(
                graph_as_json=graph_as_json_str,
                rotation=rotation_origin,
                utm_origin=origin,
                current_location="",
                flip_coords=False,
            )
            # print(f"could parse graph: {success}")

            existing_nodes = self.graph.graph.nodes
            # print(f"existing nodes: {existing_nodes}")
            existing_edges = [tuple(e) for e in self.graph.graph.edges]
            new_nodes = []
            new_edges = []

            for node in tmp_handler.graph.nodes:
                if node not in existing_nodes:
                    # print(f"new node: {node}")
                    new_nodes.append(
                        dict(
                            name=node,
                            **tmp_handler.graph.nodes[node],
                            source="from_uav",
                        )
                    )

            for edge in tmp_handler.graph.edges:
                if tuple(edge) not in existing_edges:
                    # print(f"new edge: {edge}")
                    new_edges.append(list(edge))

            for node in new_nodes:
                name = node["name"]
                attrs = node.copy()
                attrs.pop("name")
                self.graph.update_with_node(node=name, edges=[], attrs=attrs)

            for edge in new_edges:
                self.graph.update_with_edge(edge)

            for node_info in new_nodes:
                node = node_info["name"]
                info = self.graph.graph.nodes[node]
                neighbors = self.graph.get_neighbors(node)
                response = self.add_node_srv_proxy(
                    get_add_node_msg(
                        node_id=node,
                        x=info["coords"][0],
                        y=info["coords"][1],
                        type=info["type"],
                        neighbors=[],
                    )
                )
                info["name"] = node
                self.llm_prompt_former.update(new_nodes=[info])

            for node_info in new_nodes:
                node = node_info["name"]
                info = self.graph.graph.nodes[node]
                neighbors = self.graph.get_neighbors(node)
                response = self.add_node_srv_proxy(
                    get_add_node_msg(
                        node_id=node,
                        x=info["coords"][0],
                        y=info["coords"][1],
                        type=info["type"],
                        neighbors=neighbors,
                    )
                )
                connections = [[node, c] for c in neighbors]
                self.llm_prompt_former.update(new_connections=connections)
            self.get_logger().info("graph trated as perception!")


        self.graph_viz.update_graph(self.graph)
        print(f"\ncurrent loc:{self.current_location}")

    def costmap_cbk(self, costmap_msg: OccupancyGrid) -> None:
        pose_in_map = costmap_msg.info.origin
        position = np.array([pose_in_map.position.x, pose_in_map.position.y])
        yaw = Rotation.from_quat(
            [
                pose_in_map.orientation.x,
                pose_in_map.orientation.y,
                pose_in_map.orientation.z,
                pose_in_map.orientation.w,
            ]
        ).as_euler("xyz")[0]
        costmap = np.array(costmap_msg.data).reshape(
            costmap_msg.info.height, costmap_msg.info.width
        )
        filtered_costmap = self.frontier_extractor.update_costmap(
            costmap=costmap,
            resolution_m_p_cell=costmap_msg.info.resolution,
            pos=position,
            yaw=yaw,
        )

        # filtered_costmap_msg = OccupancyGrid()
        # filtered_costmap_msg.header = costmap_msg.header
        # filtered_costmap_msg.info = costmap_msg.info
        # filtered_costmap.map[filtered_costmap.map > 0] = 100
        # filtered_costmap_msg.data = filtered_costmap.map.reshape(-1)
        # self.filtered_costmap_pub.publish(filtered_costmap_msg)

        filtered_costmap_msg = OccupancyGrid()
        filtered_costmap_msg.header = costmap_msg.header
        filtered_costmap_msg.info = costmap_msg.info

        # Make sure values are valid occupancy values
        m = np.array(filtered_costmap.map, dtype=np.int16)   # or filtered_costmap if it's already an array
        m[m > 0] = 100                                       # keep unknown (-1) as-is if present
        m = np.clip(m, -1, 100)                              # safety

        filtered_costmap_msg.data = m.astype(np.int8).flatten().tolist()
        self.filtered_costmap_pub.publish(filtered_costmap_msg)

    def add_frontiers_to_graph(
        self, frontiers: np.ndarray, debug: Optional[bool] = False
    ) -> Tuple[List[FrontierNode], bool]:
        """Compute frontiers and add them to graph.

        Parameters
        ----------
        debug : Optional[bool], optional
            If true, don't actually update graph. Just get
            update message, by default False

        Returns
        -------
        - Update message in LLM API.
        - frontier (assumes one currently)
        - is frontier at obstacle boundary
        """
        for frontier in frontiers:
            region_id = frontier.id
            region_loc = frontier.location
            neighbor_ids = frontier.neighbors

            if not debug:
                self.graph.update_with_node(
                    node=region_id,
                    edges=neighbor_ids,
                    attrs={"coords": region_loc, "type": "region"},
                )

                response = self.add_node_srv_proxy(
                    get_add_node_msg(
                        node_id=region_id,
                        x=region_loc[0],
                        y=region_loc[1],
                        type="region",
                        neighbors=neighbor_ids,
                    )
                )
                assert response.success

            new_node = {
                "name": region_id,
                "type": "region",
                "coords": f"[{region_loc[0]:0.1f}, {region_loc[1]:0.1f}]",
            }
            new_connections = [[region_id, c] for c in neighbor_ids]
            self.llm_prompt_former.update(
                new_nodes=[new_node], new_connections=new_connections
            )

        self.graph_viz.update_graph(self.graph)

    def try_add_edges(self, node_id, coords, node_type) -> Tuple[List[str], List[str]]:
        new_neighbors = self.frontier_extractor.get_missing_neighbors(node_id, coords)

        if len(new_neighbors) == 0:
            return []

        all_neighbors = new_neighbors + self.graph.get_neighbors(node_id)

        # updates graph nav server and graph handler object
        self.update_node(
            node_id=node_id, coords=coords, node_type=node_type, neighbors=all_neighbors
        )

        new_connections = [[node_id, neighbor] for neighbor in new_neighbors]
        self.llm_prompt_former.update(new_connections=new_connections)
        return new_neighbors

    def update_node(self, node_id, coords, node_type, neighbors, attrs={}) -> None:
        attrs["coords"] = coords
        attrs["type"] = node_type

        self.graph.update_with_node(node=node_id, edges=neighbors, attrs=attrs)

        response = self.add_node_srv_proxy(
            get_add_node_msg(
                node_id=node_id,
                x=coords[0],
                y=coords[1],
                type=node_type,
                neighbors=neighbors,
            )
        )

    def track_cbk(self, track_msg: Track) -> None:
        # self.track_queue.put(track_msg)
        # self.clear_track_queue()

        
        parent = self.current_location

        track = from_track_msg(track_msg, parent=parent)
        

        if track.idx in self.tracks:
            if not self.tracks[track.idx].is_same(track, pos_tol=1):
                self.tracks[track.idx] = track
                self.updated_tracks.add(track.idx)
                # self.get_logger().info("track updated")
        else:
            self.tracks[track.idx] = track
            self.added_tracks.add(track.idx)
            # self.get_logger().info(f"added track: {track}")

    # TODO buggy
    def clear_track_queue(self):
        while self.track_queue.not_empty:
            track_msg = self.track_queue.get()

            # # TODO unclear if we should use closest position or
            # # current location as parent
            # track_point = np.array(
            #     [track_msg.pose.pose.position.x, track_msg.pose.pose.position.y]
            # )
            ## region_nodes, region_node_locs = self.graph.get_region_nodes_and_locs()
            # closest_region_idx = np.linalg.norm(
            #     track_point - region_node_locs[:, :2], axis=-1
            # ).argmin()
            # parent = region_nodes[closest_region_idx]
            parent = self.current_location

            track = from_track_msg(track_msg, parent=parent)

            if track.idx in self.tracks:
                if not self.tracks[track.idx].is_same(track, pos_tol=1):
                    self.tracks[track.idx] = track
                    self.updated_tracks.add(track.idx)
            else:
                self.tracks[track.idx] = track
                self.added_tracks.add(track.idx)
                self.get_logger().info(f"added track: {track}")

    def nav_to_region_cbk(self, task: Task.Request, resp: Task.Response) -> Task.Response:
        goal = task.task

        assert self.graph.contains_node(goal)

        success = self.graph_nav_to_region(goal)
        resp.success = bool(success)
        resp.message = ""

        return resp

    def graph_nav_to_region(self, goal_region: str) -> bool:
        if self.current_location == goal_region:
            return True

        # if a current path ends up being blocked, keep trying until there
        # are no more paths to exhaust
        current_iter = 0
        while self.graph.path_exists_from_current_loc(goal_region):
            path = self.graph.get_path(self.current_location, goal_region)
            self.get_logger().info(f"navigating along path: {path}")

            nav_success = True
            for node in path:
                if self.current_location == node:
                    continue
                response = self.navigate_to_region_srv_proxy.call_async(str(node)) #### Service what type it 
                nav_success = response.success

                self.get_logger().info(f"step {node} in {path} successful: {nav_success}")

                if nav_success:
                    self._update_location(node)
                else:
                    self.graph.remove_edge(self.current_location, node)
                    self.llm_prompt_former.update(
                        removed_connections=[[self.current_location, node]]
                    )

                    break

            # try to return to last known location
            if not nav_success:
                self.get_logger().info(
                    f"could not traverse path. returning to {self.current_location}"
                )
                return_response = self.navigate_to_region_srv_proxy(
                    self.current_location
                )

                if not return_response.success:
                    self.llm_prompt_former.update(
                        freeform_updates=[
                            f"could not return to {self.current_location}. Robot is likely stuck. Recommend stopping task."
                        ]
                    )

            print(f"final result of path: {path} success: {nav_success}")
            print(
                f"more paths: {self.graph.path_exists_from_current_loc(goal_region)}, current iter: {current_iter}"
            )

            # TODO logic is hacky, but timeout just in case this gets stuck in a loop
            # if nav success is true here, it means we got to the goal so we can break
            current_iter += 1
            if nav_success or current_iter >= 3:
                break

        # suggest exploration if there is no path in teh graph.
        if not nav_success:
            self.llm_prompt_former.update(
                freeform_updates=[
                    f"could not navigate between [{self.current_location}, {node}]. Connection is likely blocked.\n"
                    f"You may consider calling `extend_map` in the direction of your target region to try and find a new path."
                ]
            )

        return nav_success

    def parse_track_updates(self) -> List[str]:
        """ This is adding the graph
        Construct new object message in the planning API. 
        Objects are newly received tracks.

        Also update graph.

        Returns
        -------
        str
            New object message in LLM API.
        """
        added_track_idx = self.added_tracks.copy()
        self.added_tracks.clear()

        if len(added_track_idx) == 0:
            return ""

        for idx in added_track_idx:
            # tracks are received on route to current_location``
            # self.tracks[idx].parent = self.current_location
            track = self.tracks[idx]

            node_id = f"discovered_{track.label}_{idx}"

            new_node = {
                "name": node_id,
                "type": "object",
                "coords": f"[{track.pose[0]:0.1f}, {track.pose[1]:0.1f}]",
            }
            new_connections = [[node_id, track.parent]]
            self.llm_prompt_former.update(
                new_nodes=[new_node], new_connections=new_connections
            )

            # add node with connection to region where it was discovered
            # only add x, y
            self.graph.update_with_node(
                node=node_id,
                edges=[track.parent],
                attrs={"coords": track.pose[:2], "type": "object"},
            )
            response = self.add_node_srv_proxy(
                get_add_node_msg(
                    node_id=node_id,
                    x=track.pose[0],
                    y=track.pose[1],
                    type="object",
                    neighbors=[track.parent],
                )
            )
            assert response.success

    def _update_location(self, new_location: str) -> bool:
        self.current_location = new_location
        self.location_history.append(new_location)
        success = self.graph.update_location(new_location)
        return success

    def inspect_object(self, node_name: str, vlm_query: str) -> Task.Response:
        nearest_region = self.graph.get_neighbors(node_name)
        assert len(
            nearest_region
        ), f"objects should only have 1 neighbor. Got: {nearest_region}"
        nearest_region = nearest_region[0]
        nav_success = self.graph_nav_to_region(nearest_region)

        if not nav_success:
            self.llm_prompt_former.update(
                freeform_updates=[
                    f"Could not inspect {node_name} because robot could not navigate "
                    f"to neighboring region {nearest_region}"
                ]
            )
            return ""

        self._update_location(nearest_region)
        response = self.inspect_object_srv_proxy(node_name)

        query = Query.Request()
        query.query = ascii(vlm_query)

        answer = self.query_scene_srv_proxy(query)
        self.llm_prompt_former.update(
            attribute_updates=[{"name": node_name, "description": answer.answer}]
        )
        # update_str = f"update_node_attributes({node_name}, description={answer.answer})"
        update_str = ""

        # assert response.success # TODO don't need this
        return update_str

    def _update_action_str(self, past_actions, new_action):
        past_actions += new_action if past_actions == "" else f", {new_action}"
        return past_actions

    def classify_region(self, region_node: str) -> str:
        """Classify region node via VLM service.

        Parameters
        ----------
        region_node : str
            Node being classified

        Returns
        -------
        str
            Updates formatted in LLM API (if any)
        """
        region_cls = self.classify_scene_srv_proxy()
        self.get_logger().info(f"scene classified to be {region_cls}")

        if region_cls != "unknown":
            self.graph.update_node_description(
                region_node, description=region_cls.message
            )

            self.llm_prompt_former.update(
                attribute_updates=[
                    {"name": region_node, "description": region_cls.message}
                ]
            )

            return ""

            # return f"update_node_attributes({region_node}, description={region_cls.message}), "

        return ""

    def execute_plan_sequence(
        self, plan: List[Tuple[str, str]]
    ) -> Tuple[bool, List[str], str]:
        """Execute one planning iteration.

        Actions in the set (goto, explore, inspect, answer) will be sequentially executed
        until a stopping condition is hit. Stopping conditions include
        - discovering a new node
        - providing an answer

        TODO this desprately needs some refactoring

        Parameters
        ----------
        plan : List[Tuple[str, str]]
            (action,  argument) pairs

        Returns
        -------
        Tuple[bool, str, str]
            - should end the task. True if the task is considered complete by the
                planner
            - update_str: perception update to be given to the llm. Uses the perception api.
            - past_actions: past actions to be given to the llm for context.
        """
        should_end = False
        should_break = False
        action_updates = []
        for action, arg in plan:
            action_updates.append(f"{action}({arg})")
            self.get_logger().info(f"executing: {action}({arg})")

            if action == "goto":
                response = self.graph_nav_to_region(arg)
                if not response:
                    should_break = True
                    break

            elif action == "map_region":
                response = self.graph_nav_to_region(arg)
                if not response:
                    should_break = True
                    break

                if self.use_sim_perception:
                    raise ValueError(f"not implemented")
                    discovered_nodes, discovered_edges = self.perception.explore(arg)

                    iteration_updates.extend(
                        self.perception.get_update_api(
                            arg, discovered_nodes, discovered_edges
                        )
                    )

                if self.should_classify_region:
                    self.classify_region(arg)

                # check for new edges
                coords, _ = self.graph.get_node_coords(arg)
                self.try_add_edges(
                    node_id=arg,
                    coords=np.array(coords),
                    node_type="region",
                )

                should_break = True

            elif action == "extend_map":
                # exploration is limited by costmap
                # if the exploration goal is beyond costmap, we wil iterate untill we reach target
                closest_frontier_dist = np.inf
                should_extend_map = True
                while should_extend_map:
                    frontiers, is_at_obstacle = self.frontier_extractor.get_frontiers(
                        proposed_frontier=arg,
                        current_location=self.current_location,
                    )

                    assert (
                        len(frontiers) <= 1
                    ), f"cannot handle more than one frontier atm"

                    if len(frontiers) == 0:
                        should_extend_map = False
                        self.llm_prompt_former.update(
                            freeform_updates=[
                                f"calling {action_updates[-1]} from {self.current_location} did not find new regions. You may try calling map_region at your current location to find missing connections. Otherwise, robot is at obstacle boundary."
                            ]
                        )

                        break

                    frontier_dist = np.linalg.norm(frontiers[0].location - arg)

                    # if there are closer discovered regions, don't add
                    regions, locs = self.graph.get_region_nodes_and_locs()

                    # TODO should be param
                    n_already_added = 0

                    for region, loc in zip(regions, locs):
                        if (
                            "discovered" in region
                            and np.linalg.norm(loc - arg) < frontier_dist
                        ):
                            n_already_added += 1

                        if n_already_added >= 2:
                            should_extend_map = False
                            should_break = True

                            self.llm_prompt_former.update(
                                freeform_updates=[
                                    f"It is unlikely calling {action_updates[-1]} will find a path to {arg}. "
                                    f"Previous actions already found all promising regions. Goal is likely unreachable. "
                                    "If this was your primary goal, you should reporting findings to user. You should report objects or region attributes that may provide reason for the blockage. "
                                    "If you have other primary goals, you should move onto those. Answering will end the task, so only call `answer()` and report findings once you are done will all tasks."
                                ]
                            )

                    if should_break:
                        break

                    # if we're getting farther away from the goal, break
                    # we do not want to add this region to the map
                    if frontier_dist < closest_frontier_dist:
                        closest_frontier_dist = frontier_dist
                    else:
                        self.get_logger().info(f"frontiers are getting farther away. breaking")
                        should_extend_map = False
                        should_break = True

                        self.llm_prompt_former.update(
                            freeform_updates=[
                                f"It is unlikely calling {action_updates[-1]} from {self.current_location} will find a path to {arg}."
                                f" Current action already found all promising regions. Recommend stopping map extension."
                            ]
                        )
                    if should_break:
                        break

                    # if frontier is at obstacle or close to target, stop exploration and add
                    # new region to graph
                    if is_at_obstacle:
                        self.get_logger().info(f"frontier is at obstacle boundary. breaking")
                        self.llm_prompt_former.update(
                            freeform_updates=[
                                f"extend_map could not reach goal location {arg} because it hit an obstacle. However, check new connections to see if the discovered region is connected another region of interest."
                            ]
                        )
                        should_extend_map = False

                    elif frontier_dist < self.min_new_frontier_dist:
                        self.get_logger().info(f"frontier is close to target. breaking")
                        should_extend_map = False
                    else:
                        self.get_logger().info(f"can explore closes to boundary. continue")

                    # add our new frontier to graph
                    self.add_frontiers_to_graph(frontiers, debug=False)

                    # finally, go to new region and  check if we've found an edge to an
                    # existing node close to the target
                    response = self.graph_nav_to_region(frontiers[0].id)

                    if self.should_classify_region:
                        self.classify_region(frontiers[0].id)

                    new_neighbors = self.try_add_edges(
                        node_id=frontiers[0].id,
                        coords=np.array(frontiers[0].location),
                        node_type="region",
                    )

                    self.get_logger().info(
                        f"discovered region: {frontiers[0].id} also connected to: {new_neighbors}"
                    )

                    for new_neighbor in new_neighbors:
                        new_neighbor_coord, _ = self.graph.get_node_coords(new_neighbor)

                        # if a neighbor is close to target, stop exploring
                        if (
                            np.linalg.norm(new_neighbor_coord - arg)
                            < self.min_new_frontier_dist
                        ):
                            self.llm_prompt_former.update(
                                new_connections=[[frontiers[0].id, new_neighbor]]
                            )

                            should_extend_map = False
                            break

                should_break = True

            elif action == "explore_region":
                region_target = arg[0]
                explore_radius = arg[1]

                response = self.graph_nav_to_region(region_target)
                if not response:
                    should_break = True
                    break

                current_location, success = self.graph.get_node_coords(
                    self.current_location
                )
                assert success
                added_regions = []

                if self.should_classify_region:
                    self.classify_region(self.current_location)

                for x in [-explore_radius, 0, explore_radius]:
                    for y in [-explore_radius, 0, explore_radius]:
                        if (x == 0 and y == 0) or (x != 0 and y != 0):
                            continue

                        exploration_target = current_location + np.array([x, y])
                        (
                            frontiers,
                            is_at_boundary,
                        ) = self.frontier_extractor.get_frontiers(
                            exploration_target,
                            current_location=self.current_location,
                        )

                        if len(frontiers) > 0:
                            self.add_frontiers_to_graph(frontiers)
                            added_regions.append(frontiers[0])

                for new_region in added_regions:
                    action_updates.append(f"map_region({new_region.id})")

                    response = self.graph_nav_to_region(new_region.id)

                    if self.should_classify_region:
                        self.classify_region(new_region.id)

                if len(added_regions) == 0:
                    self.llm_prompt_former.update(
                        freeform_updates=[
                            f"Calling explore_region({current_location}) did not yield any updates. "
                            f"Either region has already been explored, or it is at a boundary. "
                        ]
                    )

                should_break = True

            # haven't implemented inspect
            elif action == "inspect":
                object_name = arg[0]
                query = arg[1]
                self.inspect_object(node_name=arg[0], vlm_query=arg[1])

                should_break = True

            # if we answer, task is complete
            elif action == "answer":
                should_end = True
                should_break = True

            elif action == "clarify":
                should_end = True
                should_break = True

            self.get_logger().info(f"at end of iteration, should_break: {should_break}")
            if should_break:
                break

            if self.should_interrupt:
                break

        # check for object updates and update graph
        if not self.use_sim_perception:
            self.parse_track_updates()

        self.graph_viz.update_graph(self.graph)

        self.action_history.extend(action_updates)
        self._action_monitor(self.action_history)

        return should_end, action_updates

    def _action_monitor(self, action_history: List[str]) -> None:
        if len(action_history) <= 1:
            return

        if action_history[-1].startswith("map_region") or action_history[-1].startswith(
            "explore_region"
        ):
            if action_history[-1] == action_history[-2]:
                self.llm_prompt_former.update(
                    freeform_updates=[
                        f"WARNING: You are calling {action_history[-1]} multiple times in a row. "
                        f"If you did not receive map updates on the first call, calling this function again "
                        f"is unlikely to yield useful information. "
                    ]
                )

        if action_history[-1].startswith("goto"):
            if action_history[-1] in action_history:
                pass

    def interrupt_cbk(self, trigger: Trigger.Response) -> Trigger.Response:
        self.should_interrupt = True
        trigger.success = True
        trigger.message = ""
        return trigger

    def set_detection_labels(self, task: str) -> bool:
        # TODO unify language

        location_description = ""
        if self.scene_description != "":
            location_description += f"The robot is {self.scene_description}. "
        location_description += "The robot has a scene graph representing its environment. These are the nodes in the graph: {nodes}"

        success, required_classes = self.class_llm.request(
            task, location_description=location_description
        )
        if not success:
            return False

        self.get_logger().info(f"Class LLM returned: {required_classes}")
        classes = required_classes["classes"]

        label_str = ",".join(classes)
        self.task_label_pub.publish(String(label_str))
        try:
            resp = self.set_detection_labels_srv(labels=label_str)
            return resp.success
        except:
            return False

    # def interrupt_cbk(self, trigger: TriggerResponse) -> TriggerResponse:
    #     self.should_interrupt = True
    #     return TriggerResponse(success=True, message="")

    def _realize_planning_iteration(
        self,
        *,
        response: Dict[str, str],
        feedback: str,
        past_actions: List[str],
        starting_loc: str,
    ) -> List[str]:
        output = (
            f"relevant graph: {response['relevant_graph']}\n"
            f"primary goal: {response['primary_goal']}\n"
            f"reasoning: {response['reasoning']}\n"
            f"plan: {response['plan']}"
        )
        self.get_logger().info(f"\n\nvalidation\n---\n{feedback}\n---")
        self.get_logger().info(f"\n\nresponse\n---\n{output}\n---")

        (
            should_end,
            action_str_list,
        ) = self.execute_plan_sequence(response["plan"])
        past_actions.extend(action_str_list)

        if starting_loc != self.current_location:
            self.llm_prompt_former.update(location_updates=[self.current_location])

        # update_str = ", ".join(update_str_list)
        if not self.should_interrupt and not should_end:
            update_str = self.llm_prompt_former.form_updates()
            action_str = ", ".join(past_actions)

            llm_prompt = f"past_actions: [{action_str}]"
            llm_prompt += f"\nupdates: [{update_str}]"
        else:
            llm_prompt = ""

        return llm_prompt, should_end, output

    def _realize_plan(self, llm_prompt: List[Dict[str, str]], starting_loc: str):
        past_actions = []
        while True:
            response, success, feedback = self.planner.request(llm_prompt)
            self.get_logger().info(str(response))
            if not success:
                return Task.Response(success=success, message=str(response))

            llm_prompt, should_end, output = self._realize_planning_iteration(
                response=response,
                feedback=feedback,
                past_actions=past_actions,
                starting_loc=starting_loc,
            )

            # plan end conditions
            if self.should_interrupt:
                self.should_interrupt = False
                self.get_logger().info(f"task interrupted. breaking")
                break
            if should_end:
                self.get_logger().info(f"task complete. breaking")
                break
            else:
                self.get_logger().info(f"will iterate with update: {llm_prompt}")

        return Task.Response(success=True, message=f"{output}")

    def task_cbk(self, task: Task.Request, resp: Task.Response) -> Task.Response:
        self.get_logger().info(f"{task.task}")
        # TODO best way to do this?
        starting_loc = copy.copy(self.current_location)

        # TODO redundant
        self.task = task.task

        # TODO do we want to reset this?
        while self.waiting_for_graph:
            self.get_logger().info(f"waiting for incoming graph")
            time.sleep(1)


        if self.use_open_vocab_detection:
            self.set_detection_labels(self.task)

        llm_prompt = f"task: {task.task}"

        # add any new updates
        update_str = self.llm_prompt_former.form_updates()
        if update_str != "":
            llm_prompt += f"\nupdates: [{update_str}]"

        print(update_str)

        return self._realize_plan(llm_prompt=llm_prompt, starting_loc=starting_loc)

    def resume_task_cbk(self, task: Task.Request, resp: Task.Response) -> Task.Response:
        self.get_logger().info(f"Resuming task")
        # query llm with last saved history
        response, success, feedback = self.planner.resume_request()
        if not success:
            resp.success = success
            resp.message = str(response)
            return resp

        # form update
        past_actions = []
        starting_loc = self.current_location
        llm_prompt, should_end, output = self._realize_planning_iteration(
            response=response,
            feedback=feedback,
            past_actions=past_actions,
            starting_loc=starting_loc,
        )

        if not success or should_end or self.should_interrupt:
            resp.success = success
            resp.message = str(response)
            return resp

        # now continue as usual
        # TODO best way to do this?
        starting_loc = copy.copy(self.current_location)
        return self._realize_plan(llm_prompt=llm_prompt, starting_loc=starting_loc)



def main():
    rclpy.init()
    planner = SPINENode()
    rclpy.spin(planner)
    planner.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()