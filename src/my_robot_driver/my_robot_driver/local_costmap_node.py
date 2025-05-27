#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration as rclpyDuration
from rclpy.time import Time as rclpyTime # Explicit import for clarity

from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PointStamped # Changed from Point to PointStamped for clarity in transformation
import tf2_ros
from tf2_ros import Buffer, TransformListener
# tf_transformations is not directly used in this script but often useful in others
# import tf_transformations
import numpy as np
import math
import tf2_geometry_msgs # For do_transform_point

class LocalCostmapNode(Node):
    def __init__(self):
        super().__init__('local_costmap_node')

        # Parameters
        self.declare_parameter('width_meters', 3.0)
        self.declare_parameter('height_meters', 3.0)
        self.declare_parameter('resolution', 0.05) # meters per cell
        self.declare_parameter('robot_base_frame', 'base_link')
        self.declare_parameter('global_frame_for_costmap', 'rtabmap_odom')
        self.declare_parameter('lidar_topic', '/scan')
        self.declare_parameter('inflation_radius_m', 0.07)
        self.declare_parameter('costmap_publish_topic', '/local_costmap/costmap')
        self.declare_parameter('costmap_update_frequency', 5.0) # Hz
        self.declare_parameter('obstacle_cost_value', 100) # Use standard Python int for declaration
        self.declare_parameter('transform_timeout_sec', 0.2) # Increased timeout slightly

        self.width_m = self.get_parameter('width_meters').value
        self.height_m = self.get_parameter('height_meters').value
        self.resolution = self.get_parameter('resolution').value
        self.robot_base_frame = self.get_parameter('robot_base_frame').value
        self.global_frame = self.get_parameter('global_frame_for_costmap').value
        self.lidar_topic_name = self.get_parameter('lidar_topic').value
        self.inflation_radius_m = self.get_parameter('inflation_radius_m').value
        self.costmap_publish_topic = self.get_parameter('costmap_publish_topic').value
        self.costmap_update_frequency = self.get_parameter('costmap_update_frequency').value
        # Get as int, then cast to np.int8 for internal use with numpy array
        self.obstacle_cost = np.int8(self.get_parameter('obstacle_cost_value').value)
        self.transform_timeout_sec = self.get_parameter('transform_timeout_sec').value

        self.width_cells = int(self.width_m / self.resolution)
        self.height_cells = int(self.height_m / self.resolution)
        self.inflation_cells = int(self.inflation_radius_m / self.resolution)

        self.costmap_data = np.full((self.height_cells, self.width_cells), 0, dtype=np.int8)

        self.tf_buffer = Buffer() # Default cache duration is 10 seconds
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.costmap_pub = self.create_publisher(OccupancyGrid, self.costmap_publish_topic, 10)
        self.scan_sub = self.create_subscription(
            LaserScan,
            self.lidar_topic_name,
            self.scan_callback,
            rclpy.qos.qos_profile_sensor_data # Appropriate for sensor data
        )
        self.latest_scan: LaserScan | None = None

        self.timer = self.create_timer(1.0 / self.costmap_update_frequency, self.update_and_publish_costmap)

        self.get_logger().info(
            f"LocalCostmapNode initialized. Global frame: '{self.global_frame}'. "
            f"Size: {self.width_m:.2f}m x {self.height_m:.2f}m ({self.width_cells}x{self.height_cells} cells), "
            f"Res: {self.resolution:.2f}m. Inflation: {self.inflation_radius_m:.2f}m ({self.inflation_cells} cells). "
            f"Publishing to {self.costmap_publish_topic}."
        )

    def scan_callback(self, msg: LaserScan):
        self.latest_scan = msg

    def update_and_publish_costmap(self):
        if self.latest_scan is None:
            self.get_logger().debug("No scan received yet.", throttle_duration_sec=5.0)
            return

        # Use rclpyTime() for "latest available" in TF lookups
        # Use self.get_clock().now().to_msg() for message headers if needed
        # For TF lookups, rclpy.time.Time() with no args means "latest"
        latest_tf_time = rclpyTime()

        try:
            # Get robot's current pose in the global_frame (e.g., rtabmap_odom)
            transform_global_to_robot = self.tf_buffer.lookup_transform(
                self.global_frame, self.robot_base_frame,
                latest_tf_time, # Request latest available for robot pose
                timeout=rclpyDuration(seconds=self.transform_timeout_sec)
            )
            robot_x_global = transform_global_to_robot.transform.translation.x
            robot_y_global = transform_global_to_robot.transform.translation.y

            # Transform for LIDAR points from lidar_frame to global_frame
            # Requesting latest available for this transform too, to try and avoid extrapolation
            transform_lidar_to_global = self.tf_buffer.lookup_transform(
                self.global_frame, self.latest_scan.header.frame_id,
                latest_tf_time, # Request latest available transform
                timeout=rclpyDuration(seconds=self.transform_timeout_sec)
            )
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f"TF lookup failed: {e}", throttle_duration_sec=1.0)
            return

        costmap_origin_x = robot_x_global - (self.width_m / 2.0)
        costmap_origin_y = robot_y_global - (self.height_m / 2.0)

        current_costmap_layer = np.full((self.height_cells, self.width_cells), 0, dtype=np.int8)

        for i, range_val in enumerate(self.latest_scan.ranges):
            if not (self.latest_scan.range_min <= range_val <= self.latest_scan.range_max):
                continue

            angle = self.latest_scan.angle_min + i * self.latest_scan.angle_increment

            p_lidar_frame = PointStamped()
            # Use the scan's timestamp for the point itself, as that's when it was measured
            p_lidar_frame.header.stamp = self.latest_scan.header.stamp
            p_lidar_frame.header.frame_id = self.latest_scan.header.frame_id
            p_lidar_frame.point.x = range_val * math.cos(angle)
            p_lidar_frame.point.y = range_val * math.sin(angle)
            p_lidar_frame.point.z = 0.0

            try:
                # Use tf2_geometry_msgs.do_transform_point
                p_global_frame = tf2_geometry_msgs.do_transform_point(p_lidar_frame, transform_lidar_to_global)
            except Exception as e: # Catch any exception during transform, including if tf2_geometry_msgs is problematic
                self.get_logger().warn(f"Failed to transform point with tf2_geometry_msgs: {e}", throttle_duration_sec=1.0)
                continue

            cell_x = int((p_global_frame.point.x - costmap_origin_x) / self.resolution)
            cell_y = int((p_global_frame.point.y - costmap_origin_y) / self.resolution)

            if 0 <= cell_x < self.width_cells and 0 <= cell_y < self.height_cells:
                current_costmap_layer[cell_y, cell_x] = self.obstacle_cost

        inflated_costmap_data = current_costmap_layer.copy()
        if self.inflation_cells > 0:
            obstacle_indices = np.argwhere(current_costmap_layer == self.obstacle_cost)
            for r_obs, c_obs in obstacle_indices:
                min_r = max(0, r_obs - self.inflation_cells)
                max_r = min(self.height_cells, r_obs + self.inflation_cells + 1)
                min_c = max(0, c_obs - self.inflation_cells)
                max_c = min(self.width_cells, c_obs + self.inflation_cells + 1)
                # Check if any part of the inflation square is within bounds before slicing
                if max_r > min_r and max_c > min_c:
                    inflated_costmap_data[min_r:max_r, min_c:max_c] = self.obstacle_cost
        
        self.costmap_data = inflated_costmap_data

        grid_msg = OccupancyGrid()
        grid_msg.header.stamp = self.get_clock().now().to_msg() # Current time for grid message
        grid_msg.header.frame_id = self.global_frame
        grid_msg.info.resolution = self.resolution
        grid_msg.info.width = self.width_cells
        grid_msg.info.height = self.height_cells
        grid_msg.info.origin.position.x = costmap_origin_x
        grid_msg.info.origin.position.y = costmap_origin_y
        grid_msg.info.origin.position.z = 0.0
        grid_msg.info.origin.orientation.w = 1.0
        grid_msg.info.origin.orientation.x = 0.0
        grid_msg.info.origin.orientation.y = 0.0
        grid_msg.info.origin.orientation.z = 0.0

        grid_msg.data = self.costmap_data.ravel().tolist()
        self.costmap_pub.publish(grid_msg)

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = LocalCostmapNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node: node.get_logger().info("LocalCostmapNode shutting down due to Ctrl-C.")
    except Exception as e:
        if node: node.get_logger().error(f"Unhandled exception in LocalCostmapNode: {e}")
        else: print(f"Unhandled exception before LocalCostmapNode init: {e}")
    finally:
        if node and rclpy.ok():
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()