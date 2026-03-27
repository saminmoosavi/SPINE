#!/usr/bin/env python3

from typing import Union

import cv2
import cv_bridge
import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CompressedImage
from std_srvs.srv import Trigger
from spine_interface_ros2.srv import Query
from spine_ros2.utility.vlm import VLMWrapper


class VLMInfer(Node):
    def __init__(self) -> None:
        super().__init__("vlm_node")

        self.declare_parameter("model", "llava-hf/vip-llava-7b-hf")
        self.declare_parameter("image_topic", "/camera/color/image_raw")
        self.declare_parameter("image_is_compressed", False)
        self.declare_parameter( "classes",
            "parking lot, road, sidewalk, park, other",
        )

        model = self.get_parameter("model").get_parameter_value().string_value
        input_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        image_is_compressed = (self.get_parameter("image_is_compressed").get_parameter_value().bool_value)
        classes_list = self.get_parameter("classes").get_parameter_value().string_value

        self.bridge = cv_bridge.CvBridge()
        self.vlm = VLMWrapper(model=model, classes=classes_list)
        self.latest_img: Union[np.ndarray, None] = None

        if image_is_compressed:
            self.img_sub = self.create_subscription(CompressedImage, input_topic, self.img_cbk, 1)
            self.get_logger().info(f"Subscribed to compressed image topic: {input_topic}")
        else:
            self.img_sub = self.create_subscription(Image, input_topic, self.img_cbk, 1)
            self.get_logger().info(f"Subscribed to raw image topic: {input_topic}")

        self.cls_scene_srv = self.create_service(Trigger,"/vlm_infer/classify_scene",self.classify_scene)
        self.open_cls_scene_srv = self.create_service(Trigger,"/vlm_infer/open_classify_scene",self.open_classify_scene)
        self.query_scene_srv = self.create_service(Query,"/vlm_infer/query_scene",self.answer_scene)

    def img_cbk(self, img_msg: Union[Image, CompressedImage]) -> None:
        try:
            self.latest_img = self.decode_img_msg(img_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to decode image: {e}")

    def classify_scene(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request

        if self.latest_img is None:
            response.success = True
            response.message = "unknown"
            return response

        msg, class_id = self.vlm.classify_scene(image=self.latest_img)
        self.get_logger().info(f"vlm returned: {msg}")

        response.success = True
        response.message = str(class_id)
        return response

    def open_classify_scene(self,request: Trigger.Request,response: Trigger.Response) -> Trigger.Response:
        del request

        if self.latest_img is None:
            response.success = True
            response.message = "unknown"
            return response

        msg, answer = self.vlm.open_classify_scene(image=self.latest_img)
        self.get_logger().info(f"vlm returned: {msg}")

        response.success = True
        response.message = str(answer)
        return response

    def answer_scene(self,request: Query.Request,response: Query.Response) -> Query.Response:
        if self.latest_img is None:
            response.success = False
            response.answer = "unknown"
            return response

        query = (
            f"{request.query}. And why? Provide a brief explanation "
            f"with details in 25 words or less."
        )
        self.get_logger().info(f"sending query: {query}")

        msg = self.vlm.open_query(prompt=query, image=self.latest_img)
        parsed = msg.split(query)[-1].strip()
        self.get_logger().info(f"vlm returned: {msg}")

        response.success = True
        response.answer = parsed
        return response

    def decode_img_msg(self, msg: Union[Image, CompressedImage]) -> np.ndarray:
        """
        Decode a ROS 2 image message into a numpy image.

        Supports:
        - sensor_msgs/msg/Image
        - sensor_msgs/msg/CompressedImage

        Returns:
            np.ndarray:
                - color images as HxWx3 RGB
                - mono/depth images as HxW
        """
        if isinstance(msg, CompressedImage):
            np_arr = np.frombuffer(msg.data, dtype=np.uint8)
            img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise ValueError("Could not decode compressed image")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            self.get_logger().debug(f"Decoded compressed image with shape {img_rgb.shape}")
            return img_rgb

        if not isinstance(msg, Image):
            raise TypeError(f"Unsupported message type: {type(msg)}")

        encoding = msg.encoding.lower()

        if encoding in ("rgb8", "bgr8", "bgra8", "rgba8", "mono8", "mono16"):
            # Use CvBridge for standard encodings
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

            if encoding == "bgr8":
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            elif encoding == "bgra8":
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2RGB)
            elif encoding == "rgba8":
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGBA2RGB)
            elif encoding in ("mono8", "mono16"):
                # leave as single channel
                pass
            elif encoding == "rgb8":
                # already RGB
                pass

            self.get_logger().debug(f"Decoded raw image encoding={msg.encoding}, shape={cv_img.shape}")
            return np.array(cv_img, copy=True)

        elif encoding == "32fc1":
            dtype = np.dtype(np.float32).newbyteorder(">" if msg.is_bigendian else "<")
            img = np.ndarray(shape=(msg.height, msg.width),dtype=dtype,buffer=msg.data).copy()
            return img

        elif encoding == "16uc1":
            dtype = np.dtype(np.uint16).newbyteorder(">" if msg.is_bigendian else "<")
            img = np.ndarray(shape=(msg.height, msg.width),dtype=dtype,buffer=msg.data).copy()
            return img

        else:
            raise ValueError(f"Unsupported image encoding: {msg.encoding}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VLMInfer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()