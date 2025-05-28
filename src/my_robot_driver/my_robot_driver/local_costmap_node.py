#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration as rclpyDuration
from rclpy.time import Time as rclpyTime

from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PointStamped
import tf2_ros
from tf2_ros import Buffer, TransformListener
import numpy as np
import math
import tf2_geometry_msgs
import collections

COST_LETHAL_PY = 100
COST_NEUTRAL_PY = 0


class LocalCostmapNode(Node):
    def __init__(self):
        super().__init__("local_costmap_node")

        self.declare_parameter("width_meters", 3.0)
        self.declare_parameter("height_meters", 3.0)
        self.declare_parameter("resolution", 0.05)
        self.declare_parameter("robot_base_frame", "base_link")
        self.declare_parameter("global_frame_for_costmap", "rtabmap_odom")
        self.declare_parameter("lidar_topic", "/scan")
        self.declare_parameter("costmap_publish_topic", "/local_costmap/costmap")
        self.declare_parameter("costmap_update_frequency", 5.0)
        self.declare_parameter("transform_timeout_sec", 0.2)
        self.declare_parameter("inflation_radius_m", 0.25)
        self.declare_parameter("max_inflation_cost", 99)
        self.declare_parameter("cost_scaling_factor", 10.0)
        self.declare_parameter("obstacle_cost_value", COST_LETHAL_PY)

        self.width_m = self.get_parameter("width_meters").value
        self.height_m = self.get_parameter("height_meters").value
        self.resolution = self.get_parameter("resolution").value
        self.robot_base_frame = self.get_parameter("robot_base_frame").value
        self.global_frame = self.get_parameter("global_frame_for_costmap").value
        self.lidar_topic_name = self.get_parameter("lidar_topic").value
        self.costmap_publish_topic = self.get_parameter("costmap_publish_topic").value
        self.costmap_update_frequency = self.get_parameter(
            "costmap_update_frequency"
        ).value
        self.transform_timeout_sec = self.get_parameter("transform_timeout_sec").value
        self.inflation_radius_m = self.get_parameter("inflation_radius_m").value
        self.max_inflation_cost_val = np.int8(
            self.get_parameter("max_inflation_cost").value
        )
        self.cost_scaling_factor = self.get_parameter("cost_scaling_factor").value
        self.sensor_obstacle_cost = np.int8(
            self.get_parameter("obstacle_cost_value").value
        )

        self.COST_LETHAL_NP = np.int8(COST_LETHAL_PY)
        self.COST_NEUTRAL_NP = np.int8(COST_NEUTRAL_PY)

        if self.sensor_obstacle_cost != self.COST_LETHAL_NP:
            self.get_logger().warn(
                f"Parameter 'obstacle_cost_value' is not COST_LETHAL. Using {self.COST_LETHAL_NP}."
            )
            self.sensor_obstacle_cost = self.COST_LETHAL_NP
        if self.max_inflation_cost_val >= self.COST_LETHAL_NP:
            self.get_logger().warn(
                f"Parameter 'max_inflation_cost' is >= COST_LETHAL. Clamping."
            )
            self.max_inflation_cost_val = np.int8(self.COST_LETHAL_NP - 1)

        self.width_cells = int(self.width_m / self.resolution)
        self.height_cells = int(self.height_m / self.resolution)
        self.inflation_radius_cells = int(self.inflation_radius_m / self.resolution)
        self.final_costmap_data = np.full(
            (self.height_cells, self.width_cells), self.COST_NEUTRAL_NP, dtype=np.int8
        )

        self.tf_buffer = Buffer()  # Default cache duration 10s, usually fine.
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.costmap_pub = self.create_publisher(
            OccupancyGrid, self.costmap_publish_topic, 10
        )
        self.scan_sub = self.create_subscription(
            LaserScan,
            self.lidar_topic_name,
            self.scan_callback,
            rclpy.qos.qos_profile_sensor_data,
        )
        self.latest_scan: LaserScan | None = None
        self.timer = self.create_timer(
            1.0 / self.costmap_update_frequency, self.update_and_publish_costmap
        )
        self.get_logger().info(
            f"LocalCostmapNode initialized. Costmap frame_id: '{self.global_frame}'."
        )

    def scan_callback(self, msg: LaserScan):
        self.latest_scan = msg

    def _compute_cost(self, distance_cells: int) -> np.int8:
        if distance_cells == 0:
            return self.sensor_obstacle_cost
        if distance_cells > self.inflation_radius_cells:
            return self.COST_NEUTRAL_NP
        factor = (1.0 - float(distance_cells) / self.inflation_radius_cells) ** 2
        cost = (
            self.max_inflation_cost_val - self.COST_NEUTRAL_NP
        ) * factor + self.COST_NEUTRAL_NP
        return np.int8(
            max(self.COST_NEUTRAL_NP, min(cost, self.max_inflation_cost_val))
        )

    def _inflate_obstacles(self, obstacle_marked_layer: np.ndarray) -> np.ndarray:
        if self.inflation_radius_cells <= 0:
            return obstacle_marked_layer
        inflated_costmap = np.copy(obstacle_marked_layer)
        queue = collections.deque()
        cell_distances = np.full(
            (self.height_cells, self.width_cells), float("inf"), dtype=float
        )
        for r in range(self.height_cells):
            for c in range(self.width_cells):
                if obstacle_marked_layer[r, c] == self.sensor_obstacle_cost:
                    queue.append((r, c, 0))
                    cell_distances[r, c] = 0
        while queue:
            curr_r, curr_c, dist = queue.popleft()
            if dist >= self.inflation_radius_cells:
                continue
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = curr_r + dr, curr_c + dc
                    if 0 <= nr < self.height_cells and 0 <= nc < self.width_cells:
                        new_dist_to_obstacle = dist + 1
                        if new_dist_to_obstacle < cell_distances[nr, nc]:
                            cell_distances[nr, nc] = new_dist_to_obstacle
                            if inflated_costmap[nr, nc] < self.sensor_obstacle_cost:
                                inflated_costmap[nr, nc] = self._compute_cost(
                                    int(new_dist_to_obstacle)
                                )
                            if new_dist_to_obstacle < self.inflation_radius_cells:
                                queue.append((nr, nc, new_dist_to_obstacle))
        return inflated_costmap

    def update_and_publish_costmap(self):
        if self.latest_scan is None:
            return

        # Use rclpyTime() with no arguments to request the "latest available" transform.
        # This is generally more robust to minor timing discrepancies.
        time_for_lookup = rclpyTime()

        try:
            # Transform to get robot's pose in the costmap's global frame
            transform_global_to_robot = self.tf_buffer.lookup_transform(
                self.global_frame,
                self.robot_base_frame,
                time_for_lookup,  # Use latest available
                timeout=rclpyDuration(seconds=self.transform_timeout_sec),
            )
            robot_x_in_global_frame = transform_global_to_robot.transform.translation.x
            robot_y_in_global_frame = transform_global_to_robot.transform.translation.y

            # Transform to get lidar points into the costmap's global frame
            transform_lidar_to_global = self.tf_buffer.lookup_transform(
                self.global_frame,
                self.latest_scan.header.frame_id,
                time_for_lookup,  # <<< USE LATEST AVAILABLE TRANSFORM
                timeout=rclpyDuration(seconds=self.transform_timeout_sec),
            )

        except Exception as e:
            self.get_logger().warn(
                f"LocalCostmap TF lookup failed: {e}", throttle_duration_sec=1.0
            )
            return

        costmap_origin_x_in_global_frame = robot_x_in_global_frame - (
            self.width_m / 2.0
        )
        costmap_origin_y_in_global_frame = robot_y_in_global_frame - (
            self.height_m / 2.0
        )

        current_sensor_obstacle_layer = np.full(
            (self.height_cells, self.width_cells), self.COST_NEUTRAL_NP, dtype=np.int8
        )

        for i, range_val in enumerate(self.latest_scan.ranges):
            if not (
                self.latest_scan.range_min <= range_val <= self.latest_scan.range_max
            ):
                continue
            angle = self.latest_scan.angle_min + i * self.latest_scan.angle_increment

            p_lidar_frame_stamped = PointStamped()
            # IMPORTANT: The point itself is defined at the scan's timestamp relative to lidar_link
            p_lidar_frame_stamped.header.stamp = self.latest_scan.header.stamp
            p_lidar_frame_stamped.header.frame_id = self.latest_scan.header.frame_id
            p_lidar_frame_stamped.point.x = range_val * math.cos(angle)
            p_lidar_frame_stamped.point.y = range_val * math.sin(angle)
            p_lidar_frame_stamped.point.z = 0.0

            try:
                # Transform this historical point using the LATEST available transform_lidar_to_global
                p_transformed_to_global_frame = tf2_geometry_msgs.do_transform_point(
                    p_lidar_frame_stamped, transform_lidar_to_global
                )
            except Exception as e:
                self.get_logger().warn(
                    f"Failed to transform lidar point: {e}", throttle_duration_sec=1.0
                )
                continue

            cell_x = int(
                (
                    p_transformed_to_global_frame.point.x
                    - costmap_origin_x_in_global_frame
                )
                / self.resolution
            )
            cell_y = int(
                (
                    p_transformed_to_global_frame.point.y
                    - costmap_origin_y_in_global_frame
                )
                / self.resolution
            )

            if 0 <= cell_x < self.width_cells and 0 <= cell_y < self.height_cells:
                current_sensor_obstacle_layer[cell_y, cell_x] = (
                    self.sensor_obstacle_cost
                )

        self.final_costmap_data = self._inflate_obstacles(current_sensor_obstacle_layer)

        grid_msg = OccupancyGrid()
        grid_msg.header.stamp = (
            self.get_clock().now().to_msg()
        )  # Costmap published with current time
        grid_msg.header.frame_id = self.global_frame
        grid_msg.info.resolution = self.resolution
        grid_msg.info.width = self.width_cells
        grid_msg.info.height = self.height_cells
        grid_msg.info.origin.position.x = costmap_origin_x_in_global_frame
        grid_msg.info.origin.position.y = costmap_origin_y_in_global_frame
        grid_msg.info.origin.position.z = 0.0
        grid_msg.info.origin.orientation.w = 1.0

        grid_msg.data = self.final_costmap_data.ravel().tolist()
        self.costmap_pub.publish(grid_msg)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = LocalCostmapNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node:
            node.get_logger().info("LocalCostmapNode shutting down due to Ctrl-C.")
    except Exception as e:
        if node:
            node.get_logger().error(
                f"Unhandled exception in LocalCostmapNode: {e}", exc_info=True
            )
        else:
            print(f"Unhandled exception before LocalCostmapNode init: {e}")
    finally:
        if node and rclpy.ok():
            if hasattr(node, "destroy_node") and callable(node.destroy_node):
                node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
