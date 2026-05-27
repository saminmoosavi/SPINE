#!/usr/bin/env python3

import json
import math


def load_world(json_file):
    with open(json_file, "r") as f:
        return json.load(f)


def get_region_coords(world, region_name):
    for region in world["regions"]:
        if region["name"] == region_name:
            return region["coords"]
    raise ValueError(f"Region not found: {region_name}")


def dist_expr(x_var, y_var, x_obj, y_obj):
    return f"sqrt(({x_var} - {x_obj})^2 + ({y_var} - {y_obj})^2)"


def point_to_segment_distance(px, py, ax, ay, bx, by):
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay

    ab_len_sq = abx**2 + aby**2

    if ab_len_sq == 0:
        return math.sqrt((px - ax)**2 + (py - ay)**2)

    t = (apx * abx + apy * aby) / ab_len_sq
    t = max(0.0, min(1.0, t))

    closest_x = ax + t * abx
    closest_y = ay + t * aby

    return math.sqrt((px - closest_x)**2 + (py - closest_y)**2)


def get_objects_near_path(world, start_xy, goal_xy, corridor_width=2.0):
    ax, ay = start_xy
    bx, by = goal_xy

    relevant_objects = []

    for obj in world["objects"]:
        ox, oy = obj["coords"]

        d = point_to_segment_distance(
            ox, oy,
            ax, ay,
            bx, by
        )

        if d <= corridor_width:
            obj_copy = obj.copy()
            obj_copy["distance_to_path"] = d
            relevant_objects.append(obj_copy)

    return relevant_objects


def generate_relevant_stl_for_task(
    json_file,
    robot_location,
    robot_goal,
    task_type="navigate",
    time_bound=120,
    goal_tolerance=0.5,
    path_corridor_width=2.0,
    human_safety_distance=1.5,
    obstacle_safety_distance=0.5,
    slow_human_distance=3.0,
    slow_obstacle_distance=1.0,
    max_speed=1.0,
    human_slow_speed=0.35,
    obstacle_slow_speed=0.25,
    max_accel=0.5,
    max_angular_speed=1.0,
):
    world = load_world(json_file)

    start_xy = get_region_coords(world, robot_location)
    goal_xy = get_region_coords(world, robot_goal)

    gx, gy = goal_xy

    relevant_objects = get_objects_near_path(
        world,
        start_xy,
        goal_xy,
        corridor_width=path_corridor_width
    )

    formulas = {
        "robot_location": robot_location,
        "robot_goal": robot_goal,
        "task_type": task_type,
        "start_xy": start_xy,
        "goal_xy": goal_xy,
        "path_corridor_width": path_corridor_width,
        "relevant_objects": relevant_objects,
        "stl": {
            "reach_avoid": [],
            "adaptive_speed": [],
            "comfort": [],
            "task_constraints": [],
        },
    }

    # ========================================================
    # 1. Reach goal
    # ========================================================

    formulas["stl"]["reach_avoid"].append({
        "name": "reach_goal",
        "formula": (
            f"F[0,{time_bound}] "
            f"({dist_expr('x', 'y', gx, gy)} <= {goal_tolerance})"
        )
    })

    # ========================================================
    # 2. Avoid only objects relevant to path
    # ========================================================

    for obj in relevant_objects:
        name = obj["name"]
        ox, oy = obj["coords"]

        if name.startswith("person"):
            safe_dist = human_safety_distance
        else:
            safe_dist = obstacle_safety_distance

        formulas["stl"]["reach_avoid"].append({
            "name": f"avoid_{name}",
            "distance_to_path": obj["distance_to_path"],
            "formula": (
                f"G[0,{time_bound}] "
                f"({dist_expr('x', 'y', ox, oy)} >= {safe_dist})"
            )
        })

    # ========================================================
    # 3. Adaptive speed only around relevant objects
    # ========================================================

    formulas["stl"]["adaptive_speed"].append({
        "name": "global_speed_limit",
        "formula": f"G[0,{time_bound}] (v <= {max_speed})",
    })

    for obj in relevant_objects:
        name = obj["name"]
        ox, oy = obj["coords"]

        if name.startswith("person"):
            slow_dist = slow_human_distance
            slow_speed = human_slow_speed
        else:
            slow_dist = slow_obstacle_distance
            slow_speed = obstacle_slow_speed

        formulas["stl"]["adaptive_speed"].append({
            "name": f"slow_near_{name}",
            "distance_to_path": obj["distance_to_path"],
            "formula": (
                f"G[0,{time_bound}] "
                f"(({dist_expr('x', 'y', ox, oy)} < {slow_dist}) "
                f"-> (v <= {slow_speed}))"
            )
        })

    # ========================================================
    # 4. Comfort formulas
    # ========================================================

    formulas["stl"]["comfort"].append({
        "name": "acceleration_limit",
        "formula": f"G[0,{time_bound}] (abs(a) <= {max_accel})",
    })

    formulas["stl"]["comfort"].append({
        "name": "angular_speed_limit",
        "formula": f"G[0,{time_bound}] (abs(w) <= {max_angular_speed})",
    })

    # ========================================================
    # 5. Task-specific constraints
    # ========================================================

    if task_type == "inspect":
        formulas["stl"]["task_constraints"].append({
            "name": "stop_during_inspection",
            "formula": f"G[0,{time_bound}] (inspection_active -> (v <= 0.05))",
        })

        formulas["stl"]["task_constraints"].append({
            "name": "inspect_only_at_goal",
            "formula": (
                f"G[0,{time_bound}] "
                f"(inspection_active -> "
                f"({dist_expr('x', 'y', gx, gy)} <= {goal_tolerance}))"
            ),
        })

    elif task_type == "pickup":
        formulas["stl"]["task_constraints"].append({
            "name": "stop_during_pickup",
            "formula": f"G[0,{time_bound}] (pickup_active -> (v <= 0.05))",
        })

        formulas["stl"]["task_constraints"].append({
            "name": "pickup_only_at_goal",
            "formula": (
                f"G[0,{time_bound}] "
                f"(pickup_active -> "
                f"({dist_expr('x', 'y', gx, gy)} <= {goal_tolerance}))"
            ),
        })

    elif task_type == "dropoff":
        formulas["stl"]["task_constraints"].append({
            "name": "stop_during_dropoff",
            "formula": f"G[0,{time_bound}] (dropoff_active -> (v <= 0.05))",
        })

        formulas["stl"]["task_constraints"].append({
            "name": "dropoff_only_at_goal",
            "formula": (
                f"G[0,{time_bound}] "
                f"(dropoff_active -> "
                f"({dist_expr('x', 'y', gx, gy)} <= {goal_tolerance}))"
            ),
        })

    return formulas


if __name__ == "__main__":
    result = generate_relevant_stl_for_task(
        json_file="src/planning_ros_pkgs/SPINE/ros/spine_ros2/data/graph.json",
        robot_location="R2",
        robot_goal="R10",
        task_type="inspect",
        path_corridor_width=2.0,
    )

    print("\nRelevant objects near path:")
    for obj in result["relevant_objects"]:
        print(
            obj["name"],
            obj["coords"],
            "distance_to_path =",
            round(obj["distance_to_path"], 2)
        )

    print("\nGenerated STL formulas:")
    for group, rules in result["stl"].items():
        print(f"\n--- {group.upper()} ---")
        for rule in rules:
            print(rule["name"])
            print(rule["formula"])