#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, MapMetaData
from geometry_msgs.msg import PointStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Empty
import numpy as np
import math
import json
import os
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs # For PointStamped transformations

DEFAULT_ZONE_COST = 100 # Standard OccupancyGrid lethal cost (0-100)

class NoGoZoneManager(Node):
    def __init__(self):
        super().__init__('no_go_zone_manager')

        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('clicked_point_topic', '/clicked_point') # RViz default
        self.declare_parameter('no_go_costmap_topic', '/no_go_zones/costmap')
        self.declare_parameter('visualization_topic', '/no_go_zones/markers')
        self.declare_parameter('global_frame', 'map') # Frame for no-go zones & output costmap
        self.declare_parameter('zone_radius_m', 0.5)
        self.declare_parameter('zone_cost_value', DEFAULT_ZONE_COST)
        self.declare_parameter('zones_config_file', os.path.expanduser('~/no_go_zones.json'))
        self.declare_parameter('auto_load_zones', True)
        self.declare_parameter('clear_zones_topic', '/no_go_zones/clear')

        self.map_topic_name = self.get_parameter('map_topic').value
        self.clicked_point_topic_name = self.get_parameter('clicked_point_topic').value
        self.no_go_costmap_topic_name = self.get_parameter('no_go_costmap_topic').value
        self.visualization_topic_name = self.get_parameter('visualization_topic').value
        self.global_frame_id = self.get_parameter('global_frame').value
        self.zone_radius_m = self.get_parameter('zone_radius_m').value
        self.zone_cost_value = np.int8(self.get_parameter('zone_cost_value').value)
        self.zones_config_file_path = self.get_parameter('zones_config_file').value
        self.auto_load_zones_on_startup = self.get_parameter('auto_load_zones').value
        self.clear_zones_topic_name = self.get_parameter('clear_zones_topic').value

        if not (0 <= self.zone_cost_value <= 100):
            self.get_logger().warn(
                f"zone_cost_value ({self.zone_cost_value}) is outside OccupancyGrid range [0, 100]. Clamping to {DEFAULT_ZONE_COST}."
            )
            self.zone_cost_value = np.int8(DEFAULT_ZONE_COST)

        self.map_metadata: MapMetaData | None = None
        self.no_go_zone_centers_world: list[tuple[float, float]] = [] # Stores (x,y) in self.global_frame_id

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        map_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.map_sub = self.create_subscription(OccupancyGrid, self.map_topic_name, self.map_callback, map_qos)
        
        self.point_sub = self.create_subscription(
            PointStamped, self.clicked_point_topic_name, self.clicked_point_callback, 10
        )
        self.clear_zones_sub = self.create_subscription(
            Empty, self.clear_zones_topic_name, self.clear_zones_callback, 10
        )

        costmap_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.costmap_pub = self.create_publisher(OccupancyGrid, self.no_go_costmap_topic_name, costmap_qos)
        self.marker_pub = self.create_publisher(MarkerArray, self.visualization_topic_name, costmap_qos) # Also use durable for markers

        if self.auto_load_zones_on_startup:
            self.load_zones_from_file()

        self.get_logger().info(
            f"NoGoZoneManager initialized. Waiting for map on '{self.map_topic_name}'. "
            f"Outputting no-go costmap to '{self.no_go_costmap_topic_name}' in '{self.global_frame_id}' frame."
        )
        self.get_logger().info(f"Click points on '{self.clicked_point_topic_name}' to add zones (radius: {self.zone_radius_m}m).")
        self.get_logger().info(f"To clear zones, publish to '{self.clear_zones_topic_name}'. "
                               f"Example: ros2 topic pub --once {self.clear_zones_topic_name} std_msgs/msg/Empty {{}}")

    def map_callback(self, msg: OccupancyGrid):
        if msg.header.frame_id != self.global_frame_id:
            self.get_logger().error(
                f"Map frame '{msg.header.frame_id}' does not match expected global_frame '{self.global_frame_id}'. "
                "No-go zones may be incorrect. Please ensure `global_frame` parameter matches the map's frame_id."
            )
            # Optionally, you could try to use msg.header.frame_id as the target if it's the first map message.
            # For now, we stick to the configured global_frame_id.
            # return # Or allow processing if TF can handle it.
        
        if self.map_metadata is None or \
           self.map_metadata.resolution != msg.info.resolution or \
           self.map_metadata.width != msg.info.width or \
           self.map_metadata.height != msg.info.height:
            self.get_logger().info(
                f"Map received/updated: {msg.info.width}x{msg.info.height}, res: {msg.info.resolution:.3f} m/cell, "
                f"frame: '{msg.header.frame_id}'"
            )
        self.map_metadata = msg.info # msg.info is MapMetaData
        # Important: Store the frame_id from the map message as that's what map_metadata refers to
        self.map_actual_frame_id = msg.header.frame_id 
        self.update_and_publish_all()

    def clicked_point_callback(self, msg: PointStamped):
        if self.map_metadata is None:
            self.get_logger().warn("Map not yet received. Cannot process clicked point for no-go zone.")
            return

        point_to_add = msg.point
        source_frame = msg.header.frame_id

        if source_frame != self.global_frame_id:
            self.get_logger().info(f"Clicked point is in frame '{source_frame}', attempting to transform to '{self.global_frame_id}'.")
            try:
                # Ensure the point has a valid timestamp for transformation
                if msg.header.stamp.sec == 0 and msg.header.stamp.nanosec == 0:
                    point_time = self.get_clock().now()
                    self.get_logger().warn(f"Clicked point has zero timestamp, using current time for transform: {point_time.nanoseconds / 1e9:.3f}s")
                else:
                    point_time = rclpy.time.Time.from_msg(msg.header.stamp)

                transform = self.tf_buffer.lookup_transform(
                    self.global_frame_id, # Target frame
                    source_frame,         # Source frame
                    point_time,           # Time of the point
                    rclpy.duration.Duration(seconds=1.0) 
                )
                transformed_point_stamped = tf2_geometry_msgs.do_transform_point(msg, transform)
                point_to_add = transformed_point_stamped.point
                self.get_logger().info(f"Transformed point to: ({point_to_add.x:.2f}, {point_to_add.y:.2f}) in '{self.global_frame_id}'.")
            except Exception as e:
                self.get_logger().error(f"Failed to transform clicked point from '{source_frame}' to '{self.global_frame_id}': {e}")
                return
        
        new_zone_center = (point_to_add.x, point_to_add.y)
        self.no_go_zone_centers_world.append(new_zone_center)
        self.get_logger().info(f"Added no-go zone at: ({new_zone_center[0]:.2f}, {new_zone_center[1]:.2f}) with radius {self.zone_radius_m:.2f}m.")
        self.save_zones_to_file()
        self.update_and_publish_all()
        
    def clear_zones_callback(self, msg: Empty):
        self.get_logger().info("Clearing all no-go zones.")
        self.no_go_zone_centers_world = []
        self.save_zones_to_file() # Save empty list
        self.update_and_publish_all()

    def update_and_publish_all(self):
        if self.map_metadata is None:
            return
        self.publish_no_go_costmap()
        self.publish_markers()

    def publish_no_go_costmap(self):
        if self.map_metadata is None:
            self.get_logger().debug("Map metadata not available, cannot publish no-go costmap.")
            return

        grid_msg = OccupancyGrid()
        grid_msg.header.stamp = self.get_clock().now().to_msg()
        # The costmap MUST be in the same frame and have the same geometry as the map it's based on.
        grid_msg.header.frame_id = self.map_actual_frame_id # Frame of the received /map
        grid_msg.info = self.map_metadata 

        costmap_data = np.full((self.map_metadata.height, self.map_metadata.width), 0, dtype=np.int8)
        
        if self.map_metadata.resolution == 0:
            self.get_logger().error("Map resolution is zero, cannot create no-go costmap.")
            return
            
        radius_cells = self.zone_radius_m / self.map_metadata.resolution

        for zx_world, zy_world in self.no_go_zone_centers_world:
            # Zone center is in self.global_frame_id. Map data is in self.map_actual_frame_id.
            # If they are different, we need to transform zone center to map_actual_frame_id.
            zx_map_frame, zy_map_frame = zx_world, zy_world
            if self.global_frame_id != self.map_actual_frame_id:
                try:
                    # Create a PointStamped for the zone center in its native global_frame_id
                    zone_center_pt_global = PointStamped()
                    zone_center_pt_global.header.frame_id = self.global_frame_id
                    zone_center_pt_global.header.stamp = self.get_clock().now().to_msg() # Use current time for transform
                    zone_center_pt_global.point.x = zx_world
                    zone_center_pt_global.point.y = zy_world
                    
                    transform_to_map_frame = self.tf_buffer.lookup_transform(
                        self.map_actual_frame_id, # Target
                        self.global_frame_id,     # Source
                        rclpy.time.Time(),        # Latest available
                        rclpy.duration.Duration(seconds=0.2)
                    )
                    transformed_pt = tf2_geometry_msgs.do_transform_point(zone_center_pt_global, transform_to_map_frame)
                    zx_map_frame, zy_map_frame = transformed_pt.point.x, transformed_pt.point.y
                except Exception as e:
                    self.get_logger().warn(f"Failed to transform no-go zone center from '{self.global_frame_id}' to map frame '{self.map_actual_frame_id}': {e}. Skipping this zone for costmap.")
                    continue
            
            map_ox, map_oy = self.map_metadata.origin.position.x, self.map_metadata.origin.position.y
            center_c = int((zx_map_frame - map_ox) / self.map_metadata.resolution)
            center_r = int((zy_map_frame - map_oy) / self.map_metadata.resolution)

            min_r = max(0, center_r - int(np.ceil(radius_cells)))
            max_r = min(self.map_metadata.height -1 , center_r + int(np.ceil(radius_cells)))
            min_c = max(0, center_c - int(np.ceil(radius_cells)))
            max_c = min(self.map_metadata.width -1, center_c + int(np.ceil(radius_cells)))

            for r_idx in range(min_r, max_r + 1):
                for c_idx in range(min_c, max_c + 1):
                    dist_sq_cells = (r_idx - center_r)**2 + (c_idx - center_c)**2
                    if dist_sq_cells <= radius_cells**2:
                        costmap_data[r_idx, c_idx] = self.zone_cost_value
        
        grid_msg.data = costmap_data.ravel().tolist()
        self.costmap_pub.publish(grid_msg)

    def publish_markers(self):
        if not hasattr(self, 'map_actual_frame_id') or self.map_actual_frame_id is None:
             # self.get_logger().debug("Map actual frame ID not set, cannot publish markers.")
             return

        marker_array = MarkerArray()
        # DELETEALL marker to clear previous ones
        delete_marker = Marker()
        delete_marker.header.frame_id = self.global_frame_id # Markers are in the frame zones were defined
        delete_marker.header.stamp = self.get_clock().now().to_msg()
        delete_marker.ns = "no_go_zones_viz"
        delete_marker.id = 0 
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)
        self.marker_pub.publish(marker_array) # Publish deleteall first

        # Add new markers
        marker_array.markers.clear() # Clear for new markers
        for i, (zx, zy) in enumerate(self.no_go_zone_centers_world):
            marker = Marker()
            marker.header.frame_id = self.global_frame_id # Frame where zones are defined
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "no_go_zones_viz"
            marker.id = i + 1 # Unique ID for each zone
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD

            marker.pose.position.x = zx
            marker.pose.position.y = zy
            marker.pose.position.z = 0.01 # Slightly above ground for visibility
            marker.pose.orientation.w = 1.0

            marker.scale.x = self.zone_radius_m * 2.0
            marker.scale.y = self.zone_radius_m * 2.0
            marker.scale.z = 0.02 # Small height

            marker.color.r = 1.0; marker.color.g = 0.0; marker.color.b = 0.0; marker.color.a = 0.4;

            marker.lifetime = rclpy.duration.Duration(seconds=0).to_msg() # Persist indefinitely
            marker_array.markers.append(marker)
        
        if marker_array.markers:
            self.marker_pub.publish(marker_array)

    def load_zones_from_file(self):
        try:
            if os.path.exists(self.zones_config_file_path):
                with open(self.zones_config_file_path, 'r') as f:
                    loaded_zones = json.load(f)
                    # Ensure it's a list of tuples/lists with 2 elements
                    self.no_go_zone_centers_world = [
                        (float(zone[0]), float(zone[1])) for zone in loaded_zones if isinstance(zone, (list, tuple)) and len(zone) == 2
                    ]
                self.get_logger().info(f"Loaded {len(self.no_go_zone_centers_world)} no-go zones from {self.zones_config_file_path}.")
            else:
                self.get_logger().info(f"No-go zone config file '{self.zones_config_file_path}' not found. Starting with an empty list.")
        except Exception as e:
            self.get_logger().error(f"Error loading no-go zones from file '{self.zones_config_file_path}': {e}")

    def save_zones_to_file(self):
        try:
            with open(self.zones_config_file_path, 'w') as f:
                json.dump(self.no_go_zone_centers_world, f, indent=2)
            self.get_logger().info(f"Saved {len(self.no_go_zone_centers_world)} no-go zones to {self.zones_config_file_path}.")
        except Exception as e:
            self.get_logger().error(f"Error saving no-go zones to file '{self.zones_config_file_path}': {e}")

def main(args=None):
    rclpy.init(args=args)
    node = NoGoZoneManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("NoGoZoneManager shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()