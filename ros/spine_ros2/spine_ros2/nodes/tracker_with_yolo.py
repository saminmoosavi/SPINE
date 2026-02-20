#!/usr/bin/env python3

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import message_filters

from yolo_msgs.msg import DetectionArray
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry

from geometry_msgs.msg import PoseWithCovariance, PointStamped

# PointCloud2 helpers (ROS 2 Python)
from sensor_msgs_py import point_cloud2 as pc2

# TF2 for frame transforms
import tf2_ros
from tf2_ros import TransformException
from tf2_geometry_msgs import do_transform_point

# >>> CHANGE THIS IMPORT TO YOUR PACKAGE <<<
# e.g. from track_msgs.msg import Track
from spine_interface_ros2.msg import Track  


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


class YoloTrackingPointCloudToTrack(Node):
    def __init__(self) -> None:
        super().__init__("yolo_tracking_pointcloud_to_track")
        self._warned = set()

        # -------- Parameters --------
        self.declare_parameter("namespace", "/a200_0000")

        self.declare_parameter("tracking_topic", "/yolo/tracking")
        self.declare_parameter("points_topic", "/sensors/camera_0/points")
        self.declare_parameter("odom_topic", "/platform/odom/filtered")
        self.declare_parameter("out_topic", "/tracks")

        # Transform points into this frame. If empty, we use Odometry header.frame_id.
        self.declare_parameter("target_frame", "")

        # If YOLO msg doesn’t include track id, fallback to detection index.
        self.declare_parameter("track_id_field_candidates", ["id", "track_id", "tracking_id"])

        # Reject bad points
        self.declare_parameter("max_range_m", 50.0)
        self.declare_parameter("min_range_m", 0.1)

        # Covariance heuristic
        self.declare_parameter("sigma_xy_base", 0.05)  # meters
        self.declare_parameter("sigma_z_base", 0.10)   # meters

        ns = self.get_parameter("namespace").value.rstrip("/")

        tracking_topic = self.get_parameter("tracking_topic").value
        points_topic = ns + self.get_parameter("points_topic").value
        odom_topic = ns + self.get_parameter("odom_topic").value
        out_topic = ns + self.get_parameter("out_topic").value

        self._target_frame_param = str(self.get_parameter("target_frame").value).strip()
        self._track_id_fields = list(self.get_parameter("track_id_field_candidates").value)

        self._min_r = float(self.get_parameter("min_range_m").value)
        self._max_r = float(self.get_parameter("max_range_m").value)

        self._sigma_xy_base = float(self.get_parameter("sigma_xy_base").value)
        self._sigma_z_base = float(self.get_parameter("sigma_z_base").value)

        # -------- QoS --------
        qos_sensor = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        qos_out = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        # -------- TF --------
        self._tf_buffer = tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=10.0))
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # -------- Pub --------
        self._pub = self.create_publisher(Track, out_topic, qos_out)

        # -------- Subs (sync tracking + points) --------
        self._sub_tracking = message_filters.Subscriber(self, DetectionArray, tracking_topic, qos_profile=qos_sensor)
        self._sub_points = message_filters.Subscriber(self, PointCloud2, points_topic, qos_profile=qos_sensor)

        # Odom is optional (not time-synced); we keep latest for fallback frame_id/time
        self._last_odom: Optional[Odometry] = None
        self._odom_sub = self.create_subscription(Odometry, odom_topic, self._odom_cb, qos_sensor)
        # self.track_sub = self.create_subscription(DetectionArray, tracking_topic, self.track_cb, qos_sensor)

        self._sync = message_filters.ApproximateTimeSynchronizer(
            [self._sub_tracking, self._sub_points],
            queue_size=20,
            slop=0.10,
            allow_headerless=False,
        )
        self._sync.registerCallback(self._cb)

        self.get_logger().info(
            f"Namespace: {ns}\n"
            f"Listening:\n"
            f"  tracking: {tracking_topic} (yolo_msgs/DetectionArray)\n"
            f"  points:   {points_topic}   (sensor_msgs/PointCloud2)\n"
            f"  odom:     {odom_topic}     (nav_msgs/Odometry)\n"
            f"Publishing:\n"
            f"  tracks:   {out_topic}      (Track)\n"
            f"Target frame param: '{self._target_frame_param or '(use odom frame_id)'}'"
        )
    
    def _warn_once(self, key: str, msg: str) -> None:
        if key in self._warned:
            return
        self._warned.add(key)
        self.get_logger().warn(msg)

    def _odom_cb(self, msg: Odometry) -> None:
        # self.get_logger().info("HIE")
        self._last_odom = msg
    # def track_cb(self, msg: DetectionArray) -> None:
    #     self.get_logger().info("HIE")
    #     self._last_track = msg

    # ---------- YOLO field extraction ----------
    def _extract_bbox_center_px(self, det) -> Optional[Tuple[int, int]]:
        """
        yolo_msgs Detection bbox formats can vary; try common patterns.
        """
        bbox = getattr(det, "bbox", None)
        if bbox is None:
            return None

        center = getattr(bbox, "center", None)
        if center is not None:
            pos = getattr(center, "position", None)
            if pos is not None and hasattr(pos, "x") and hasattr(pos, "y"):
                return int(round(pos.x)), int(round(pos.y))
            if hasattr(center, "x") and hasattr(center, "y"):
                return int(round(center.x)), int(round(center.y))

        # fallback: xmin/xmax/ymin/ymax
        if all(hasattr(bbox, a) for a in ["xmin", "xmax", "ymin", "ymax"]):
            cx = 0.5 * (float(bbox.xmin) + float(bbox.xmax))
            cy = 0.5 * (float(bbox.ymin) + float(bbox.ymax))
            return int(round(cx)), int(round(cy))

        # fallback: x_offset/y_offset/width/height
        if all(hasattr(bbox, a) for a in ["x_offset", "y_offset", "width", "height"]):
            cx = float(bbox.x_offset) + 0.5 * float(bbox.width)
            cy = float(bbox.y_offset) + 0.5 * float(bbox.height)
            return int(round(cx)), int(round(cy))

        return None

    def _extract_track_id(self, det, fallback_idx: int) -> int:
        for name in self._track_id_fields:
            if hasattr(det, name):
                try:
                    return int(getattr(det, name))
                except Exception:
                    pass
        return fallback_idx

    def _extract_class_id_label(self, det) -> Tuple[int, str]:
        class_id = -1
        label = ""

        for cid_field in ["class_id", "class", "id"]:
            if hasattr(det, cid_field):
                try:
                    class_id = int(getattr(det, cid_field))
                    break
                except Exception:
                    pass

        for lbl_field in ["label", "class_name", "name"]:
            if hasattr(det, lbl_field):
                try:
                    label = str(getattr(det, lbl_field))
                    break
                except Exception:
                    pass

        # nested fallback
        if (class_id == -1 or label == "") and hasattr(det, "results"):
            try:
                results = det.results
                if len(results) > 0:
                    r0 = results[0]
                    if class_id == -1:
                        for f in ["id", "class_id"]:
                            if hasattr(r0, f):
                                class_id = int(getattr(r0, f))
                                break
                    if label == "":
                        for f in ["class_name", "label", "name"]:
                            if hasattr(r0, f):
                                label = str(getattr(r0, f))
                                break
            except Exception:
                pass

        return class_id, label

    # ---------- PointCloud2 sampling ----------

    def _xyz_from_organized_cloud(self, cloud: PointCloud2, u: int, v: int):
        w = int(cloud.width)
        h = int(cloud.height)

        if h <= 1:
            self.get_logger().warn_once(
                "PointCloud2 appears UNORGANIZED (height<=1). This method needs organized cloud."
            )
            return None

        u = clamp(u, 0, w - 1)
        v = clamp(v, 0, h - 1)

        # Build numpy dtype from PointCloud2 fields (handles padding/offsets)
        try:
            dtype = pc2.dtype_from_fields(cloud.fields, point_step=cloud.point_step)
            arr = np.frombuffer(cloud.data, dtype=dtype)
        except Exception as e:
            self.get_logger().warn(f"Failed creating numpy view of PointCloud2: {e}")
            return None

        # Handle endianness if needed
        if cloud.is_bigendian:
            arr = arr.byteswap().newbyteorder()

        # Reshape to organized image: (height, width)
        try:
            arr = arr.reshape((h, w))
        except Exception as e:
            self.get_logger().warn(f"Failed reshaping PointCloud2 to (h,w)=({h},{w}): {e}")
            return None

        # Extract x,y,z (field names must exist)
        if "x" not in arr.dtype.names or "y" not in arr.dtype.names or "z" not in arr.dtype.names:
            self.get_logger().warn_once(f"PointCloud2 missing x/y/z fields. Has: {arr.dtype.names}")
            return None

        x = float(arr["x"][v, u])
        y = float(arr["y"][v, u])
        z = float(arr["z"][v, u])

        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            return None

        r = math.sqrt(x*x + y*y + z*z)
        if not (self._min_r <= r <= self._max_r):
            return None

        return x, y, z



    # ---------- Pose + covariance ----------
    def _make_pose_with_cov(self, x: float, y: float, z: float) -> PoseWithCovariance:
        p = PoseWithCovariance()
        p.pose.position.x = x
        p.pose.position.y = y
        p.pose.position.z = z

        # No orientation estimate from a bbox; identity quaternion
        p.pose.orientation.w = 1.0

        # Range-based uncertainty heuristic
        rng = max(0.001, math.sqrt(x * x + y * y + z * z))
        sigma_xy = max(self._sigma_xy_base, 0.02 * rng)
        sigma_z = max(self._sigma_z_base, 0.05 * rng)

        cov = [0.0] * 36
        cov[0] = sigma_xy ** 2  # x
        cov[7] = sigma_xy ** 2  # y
        cov[14] = sigma_z ** 2  # z

        # big unknown orientation
        cov[21] = (math.radians(30.0) ** 2)
        cov[28] = (math.radians(30.0) ** 2)
        cov[35] = (math.radians(60.0) ** 2)
        p.covariance = cov
        return p

    def _transform_point_if_needed(
        self,
        xyz: Tuple[float, float, float],
        src_frame: str,
        stamp,
        target_frame: str,
    ) -> Tuple[Tuple[float, float, float], str]:
        """
        Attempt TF transform src_frame -> target_frame at time stamp.
        If TF missing, return original xyz + src_frame.
        """
        if not target_frame or target_frame == src_frame:
            return xyz, src_frame

        pt = PointStamped()
        pt.header.stamp = stamp
        pt.header.frame_id = src_frame
        pt.point.x, pt.point.y, pt.point.z = xyz

        try:
            tf = self._tf_buffer.lookup_transform(
                target_frame,
                src_frame,
                stamp,
                timeout=rclpy.duration.Duration(seconds=0.05),
            )
            pt_out = do_transform_point(pt, tf)
            return (pt_out.point.x, pt_out.point.y, pt_out.point.z), target_frame
        except TransformException as e:
            # Do NOT crash—just fall back
            self._warn_once(
                f"tf_missing_{src_frame}_to_{target_frame}",
                f"TF missing {src_frame} -> {target_frame}: {e}. Publishing in {src_frame}."
            )
            return xyz, src_frame

    # ---------- Main callback ----------
    def _cb(self, tracking_msg: DetectionArray, cloud_msg: PointCloud2) -> None:
        # Choose target frame: param > odom.header.frame_id > keep cloud frame
        odom_frame = self._last_odom.header.frame_id if self._last_odom is not None else ""
        target_frame = self._target_frame_param or odom_frame

        cloud_frame = cloud_msg.header.frame_id
        stamp = tracking_msg.header.stamp  # align output time to detection time

        # Get detection list field (yolo_msgs usually uses `detections`)
        dets = None
        for f in ["detections", "results", "tracks"]:
            if hasattr(tracking_msg, f):
                dets = getattr(tracking_msg, f)
                break
        if dets is None:
            self.get_logger().warn("DetectionArray has no detections list field (expected detections/results/tracks).")
            return

        for i, det in enumerate(dets):
            center = self._extract_bbox_center_px(det)
            if center is None:
                continue
            u, v = center

            xyz = self._xyz_from_organized_cloud(cloud_msg, u, v)
            if xyz is None:
                continue

            xyz_out, used_frame = self._transform_point_if_needed(
                xyz=xyz,
                src_frame=cloud_frame,
                stamp=stamp,
                target_frame=target_frame,
            )

            track_id = self._extract_track_id(det, fallback_idx=i)
            class_id, label = self._extract_class_id_label(det)

            out = Track()
            out.header = tracking_msg.header
            out.header.frame_id = used_frame  # frame of the pose we publish
            out.idx = int(track_id)
            out.class_id = int(class_id)
            out.label = str(label)
            out.pose = self._make_pose_with_cov(*xyz_out)
            # self.get_logger().info("HII")
            self._pub.publish(out)


def main() -> None:
    rclpy.init()
    node = YoloTrackingPointCloudToTrack()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
