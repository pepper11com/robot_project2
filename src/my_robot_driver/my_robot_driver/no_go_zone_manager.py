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

# # To clear all zones:
# ros2 topic pub --once /no_go_zones/clear std_msgs/msg/Empty {}
# # To complete current polygon:
# ros2 topic pub --once /no_go_zones/complete_polygon std_msgs/msg/Empty {}
# # To delete last zone:
# ros2 topic pub --once /no_go_zones/delete_last std_msgs/msg/Empty {}

DEFAULT_ZONE_COST = 100 # Standard OccupancyGrid lethal cost (0-100)

class NoGoZoneManager(Node):
    def __init__(self):
        super().__init__('no_go_zone_manager')

        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('clicked_point_topic', '/clicked_point') # RViz default
        self.declare_parameter('no_go_costmap_topic', '/no_go_zones/costmap')
        self.declare_parameter('visualization_topic', '/no_go_zones/markers')
        self.declare_parameter('global_frame', 'map') # Frame for no-go zones & output costmap
        self.declare_parameter('zone_cost_value', DEFAULT_ZONE_COST)
        self.declare_parameter('zone_inflation_radius_m', 0.2)  # Add inflation around zones
        self.declare_parameter('zones_config_file', os.path.expanduser('~/no_go_zones.json'))
        self.declare_parameter('auto_load_zones', True)
        self.declare_parameter('clear_zones_topic', '/no_go_zones/clear')
        self.declare_parameter('complete_polygon_topic', '/no_go_zones/complete_polygon')
        self.declare_parameter('min_polygon_points', 3)
        self.declare_parameter('double_click_timeout_s', 1.0)
        self.declare_parameter('delete_last_zone_topic', '/no_go_zones/delete_last')

        self.map_topic_name = self.get_parameter('map_topic').value
        self.clicked_point_topic_name = self.get_parameter('clicked_point_topic').value
        self.no_go_costmap_topic_name = self.get_parameter('no_go_costmap_topic').value
        self.visualization_topic_name = self.get_parameter('visualization_topic').value
        self.global_frame_id = self.get_parameter('global_frame').value
        self.zone_cost_value = np.int8(self.get_parameter('zone_cost_value').value)
        self.zone_inflation_radius_m = self.get_parameter('zone_inflation_radius_m').value
        self.zones_config_file_path = self.get_parameter('zones_config_file').value
        self.auto_load_zones_on_startup = self.get_parameter('auto_load_zones').value
        self.clear_zones_topic_name = self.get_parameter('clear_zones_topic').value
        self.complete_polygon_topic_name = self.get_parameter('complete_polygon_topic').value
        self.min_polygon_points = self.get_parameter('min_polygon_points').value
        self.double_click_timeout_s = self.get_parameter('double_click_timeout_s').value
        self.delete_last_zone_topic_name = self.get_parameter('delete_last_zone_topic').value

        # Force zone cost to be lethal
        if self.zone_cost_value < 100:
            self.get_logger().warn(f"zone_cost_value ({self.zone_cost_value}) < 100. Setting to 100 for truly impassable zones.")
            self.zone_cost_value = np.int8(100)

        self.map_metadata: MapMetaData | None = None
        # Change from single points to list of polygons
        # Each polygon is a list of (x, y) points in self.global_frame_id
        self.no_go_polygons: list[list[tuple[float, float]]] = []
        
        # State for building current polygon
        self.current_polygon_points: list[tuple[float, float]] = []
        self.last_click_time = None
        self.is_building_polygon = False

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
        self.complete_polygon_sub = self.create_subscription(
            Empty, self.complete_polygon_topic_name, self.complete_polygon_callback, 10
        )
        self.delete_last_sub = self.create_subscription(
            Empty, self.delete_last_zone_topic_name, self.delete_last_zone_callback, 10
        )

        costmap_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.costmap_pub = self.create_publisher(OccupancyGrid, self.no_go_costmap_topic_name, costmap_qos)
        self.marker_pub = self.create_publisher(MarkerArray, self.visualization_topic_name, costmap_qos)

        if self.auto_load_zones_on_startup:
            self.load_zones_from_file()

        self.get_logger().info(
            f"NoGoZoneManager (Polygon Mode) initialized. Waiting for map on '{self.map_topic_name}'."
        )
        self.get_logger().info(
            f"Click points on '{self.clicked_point_topic_name}' to build polygons. "
            f"Double-click or publish to '{self.complete_polygon_topic_name}' to complete polygon."
        )
        self.get_logger().info(f"To clear all zones: ros2 topic pub --once {self.clear_zones_topic_name} std_msgs/msg/Empty {{}}")
        self.get_logger().info(f"To complete current polygon: ros2 topic pub --once {self.complete_polygon_topic_name} std_msgs/msg/Empty {{}}")
        self.get_logger().info(f"To delete last zone: ros2 topic pub --once {self.delete_last_zone_topic_name} std_msgs/msg/Empty {{}}")

    def map_callback(self, msg: OccupancyGrid):
        if msg.header.frame_id != self.global_frame_id:
            self.get_logger().error(
                f"Map frame '{msg.header.frame_id}' does not match expected global_frame '{self.global_frame_id}'. "
                "No-go zones may be incorrect. Please ensure `global_frame` parameter matches the map's frame_id."
            )
        
        if self.map_metadata is None or \
           self.map_metadata.resolution != msg.info.resolution or \
           self.map_metadata.width != msg.info.width or \
           self.map_metadata.height != msg.info.height:
            self.get_logger().info(
                f"Map received/updated: {msg.info.width}x{msg.info.height}, res: {msg.info.resolution:.3f} m/cell, "
                f"frame: '{msg.header.frame_id}'"
            )
        self.map_metadata = msg.info
        self.map_actual_frame_id = msg.header.frame_id 
        self.update_and_publish_all()

    def clicked_point_callback(self, msg: PointStamped):
        if self.map_metadata is None:
            self.get_logger().warn("Map not yet received. Cannot process clicked point for no-go zone.")
            return

        point_to_add = msg.point
        source_frame = msg.header.frame_id

        # Transform point to global frame if needed
        if source_frame != self.global_frame_id:
            try:
                if msg.header.stamp.sec == 0 and msg.header.stamp.nanosec == 0:
                    point_time = self.get_clock().now()
                else:
                    point_time = rclpy.time.Time.from_msg(msg.header.stamp)

                transform = self.tf_buffer.lookup_transform(
                    self.global_frame_id,
                    source_frame,
                    point_time,
                    rclpy.duration.Duration(seconds=1.0) 
                )
                transformed_point_stamped = tf2_geometry_msgs.do_transform_point(msg, transform)
                point_to_add = transformed_point_stamped.point
            except Exception as e:
                self.get_logger().error(f"Failed to transform clicked point from '{source_frame}' to '{self.global_frame_id}': {e}")
                return
        
        new_point = (point_to_add.x, point_to_add.y)
        current_time = self.get_clock().now()
        
        # Check for double-click to complete polygon
        is_double_click = False
        if (self.last_click_time is not None and 
            (current_time - self.last_click_time).nanoseconds / 1e9 < self.double_click_timeout_s and
            len(self.current_polygon_points) >= self.min_polygon_points):
            is_double_click = True
        
        self.last_click_time = current_time
        
        if is_double_click:
            self.complete_current_polygon()
        else:
            # Add point to current polygon
            self.current_polygon_points.append(new_point)
            self.is_building_polygon = True
            self.get_logger().info(
                f"Added point {len(self.current_polygon_points)}: ({new_point[0]:.2f}, {new_point[1]:.2f}). "
                f"Need {max(0, self.min_polygon_points - len(self.current_polygon_points))} more points minimum."
            )
            if len(self.current_polygon_points) >= self.min_polygon_points:
                self.get_logger().info("Double-click or publish to complete_polygon_topic to finish this polygon.")
        
        self.update_and_publish_all()

    def complete_polygon_callback(self, msg: Empty):
        self.complete_current_polygon()

    def complete_current_polygon(self):
        if len(self.current_polygon_points) < self.min_polygon_points:
            self.get_logger().warn(f"Cannot complete polygon: need at least {self.min_polygon_points} points, have {len(self.current_polygon_points)}.")
            return
        
        # Add completed polygon to list
        self.no_go_polygons.append(self.current_polygon_points.copy())
        self.get_logger().info(f"Completed polygon with {len(self.current_polygon_points)} points. Total polygons: {len(self.no_go_polygons)}")
        
        # Reset for next polygon
        self.current_polygon_points = []
        self.is_building_polygon = False
        self.last_click_time = None
        
        self.save_zones_to_file()
        self.update_and_publish_all()
        
    def clear_zones_callback(self, msg: Empty):
        self.get_logger().info("Clearing all no-go zones and current polygon.")
        self.no_go_polygons = []
        self.current_polygon_points = []
        self.is_building_polygon = False
        self.last_click_time = None
        self.save_zones_to_file()
        self.update_and_publish_all()

    def delete_last_zone_callback(self, msg: Empty):
        if self.is_building_polygon and self.current_polygon_points:
            # If currently building a polygon, cancel it
            self.get_logger().info(f"Canceling current polygon with {len(self.current_polygon_points)} points.")
            self.current_polygon_points = []
            self.is_building_polygon = False
            self.last_click_time = None
        elif self.no_go_polygons:
            # Delete the last completed polygon
            deleted_polygon = self.no_go_polygons.pop()
            self.get_logger().info(f"Deleted last polygon with {len(deleted_polygon)} points. Remaining polygons: {len(self.no_go_polygons)}")
            self.save_zones_to_file()
        else:
            self.get_logger().info("No polygons to delete.")
        
        self.update_and_publish_all()

    def _point_in_polygon(self, x: float, y: float, polygon: list[tuple[float, float]]) -> bool:
        """Ray casting algorithm to check if point is inside polygon."""
        if len(polygon) < 3:
            return False
        
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside

    def update_and_publish_all(self):
        if self.map_metadata is None:
            return
        self.publish_no_go_costmap()
        self.publish_markers()

    def publish_no_go_costmap(self):
        if self.map_metadata is None:
            return

        grid_msg = OccupancyGrid()
        grid_msg.header.stamp = self.get_clock().now().to_msg()
        grid_msg.header.frame_id = self.map_actual_frame_id
        grid_msg.info = self.map_metadata 

        costmap_data = np.full((self.map_metadata.height, self.map_metadata.width), 0, dtype=np.int8)
        
        if self.map_metadata.resolution == 0:
            self.get_logger().error("Map resolution is zero, cannot create no-go costmap.")
            return

        # Process completed polygons
        for polygon in self.no_go_polygons:
            self._rasterize_polygon(polygon, costmap_data)
        
        grid_msg.data = costmap_data.ravel().tolist()
        self.costmap_pub.publish(grid_msg)

    def _rasterize_polygon(self, polygon_world: list[tuple[float, float]], costmap_data: np.ndarray):
        """Fill polygon area in costmap with zone cost and add inflation."""
        if len(polygon_world) < 3:
            return
        
        # Transform polygon to map frame if needed
        polygon_map_frame = []
        for px_world, py_world in polygon_world:
            px_map, py_map = px_world, py_world
            if self.global_frame_id != self.map_actual_frame_id:
                try:
                    point_global = PointStamped()
                    point_global.header.frame_id = self.global_frame_id
                    point_global.header.stamp = self.get_clock().now().to_msg()
                    point_global.point.x = px_world
                    point_global.point.y = py_world
                    
                    transform = self.tf_buffer.lookup_transform(
                        self.map_actual_frame_id,
                        self.global_frame_id,
                        rclpy.time.Time(),
                        rclpy.duration.Duration(seconds=0.2)
                    )
                    transformed_pt = tf2_geometry_msgs.do_transform_point(point_global, transform)
                    px_map, py_map = transformed_pt.point.x, transformed_pt.point.y
                except Exception as e:
                    self.get_logger().warn(f"Failed to transform polygon point: {e}")
                    continue
            polygon_map_frame.append((px_map, py_map))
        
        if len(polygon_map_frame) < 3:
            return
        
        # Calculate inflation radius in cells
        inflation_cells = int(self.zone_inflation_radius_m / self.map_metadata.resolution) if self.map_metadata.resolution > 0 else 0
        
        # Find bounding box in cell coordinates (expanded for inflation)
        min_x = min(p[0] for p in polygon_map_frame) - self.zone_inflation_radius_m
        max_x = max(p[0] for p in polygon_map_frame) + self.zone_inflation_radius_m
        min_y = min(p[1] for p in polygon_map_frame) - self.zone_inflation_radius_m
        max_y = max(p[1] for p in polygon_map_frame) + self.zone_inflation_radius_m
        
        map_ox, map_oy = self.map_metadata.origin.position.x, self.map_metadata.origin.position.y
        
        min_c = max(0, int((min_x - map_ox) / self.map_metadata.resolution))
        max_c = min(self.map_metadata.width - 1, int((max_x - map_ox) / self.map_metadata.resolution))
        min_r = max(0, int((min_y - map_oy) / self.map_metadata.resolution))
        max_r = min(self.map_metadata.height - 1, int((max_y - map_oy) / self.map_metadata.resolution))
        
        # Check each cell in expanded bounding box
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                # Convert cell center to world coordinates
                world_x = map_ox + (c + 0.5) * self.map_metadata.resolution
                world_y = map_oy + (r + 0.5) * self.map_metadata.resolution
                
                if self._point_in_polygon(world_x, world_y, polygon_map_frame):
                    # Inside polygon - mark as lethal
                    costmap_data[r, c] = self.zone_cost_value
                elif inflation_cells > 0:
                    # Check if within inflation distance of polygon
                    min_dist = self._min_distance_to_polygon(world_x, world_y, polygon_map_frame)
                    if min_dist <= self.zone_inflation_radius_m:
                        # Apply inflation cost (high but not lethal)
                        inflation_cost = max(95, int(100 - (min_dist / self.zone_inflation_radius_m) * 5))
                        costmap_data[r, c] = max(costmap_data[r, c], inflation_cost)

    def _min_distance_to_polygon(self, px: float, py: float, polygon: list[tuple[float, float]]) -> float:
        """Calculate minimum distance from point to polygon edges."""
        min_dist = float('inf')
        
        for i in range(len(polygon)):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % len(polygon)]
            
            dist = self._point_to_line_distance(px, py, p1[0], p1[1], p2[0], p2[1])
            min_dist = min(min_dist, dist)
        
        return min_dist

    def _point_to_line_distance(self, px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
        """Calculate distance from point to line segment."""
        # Vector from line start to point
        dx = px - x1
        dy = py - y1
        
        # Vector of line segment
        lx = x2 - x1
        ly = y2 - y1
        
        # Length squared of line segment
        len_sq = lx * lx + ly * ly
        
        if len_sq < 1e-10:  # Line is a point
            return math.sqrt(dx * dx + dy * dy)
        
        # Project point onto line
        t = max(0, min(1, (dx * lx + dy * ly) / len_sq))
        
        # Closest point on line segment
        closest_x = x1 + t * lx
        closest_y = y1 + t * ly
        
        # Distance to closest point
        return math.sqrt((px - closest_x) ** 2 + (py - closest_y) ** 2)

    def publish_markers(self):
        if not hasattr(self, 'map_actual_frame_id') or self.map_actual_frame_id is None:
             return

        marker_array = MarkerArray()
        
        # Delete all previous markers
        delete_marker = Marker()
        delete_marker.header.frame_id = self.global_frame_id
        delete_marker.header.stamp = self.get_clock().now().to_msg()
        delete_marker.ns = "no_go_polygons"
        delete_marker.id = 0 
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)
        self.marker_pub.publish(marker_array)

        marker_array.markers.clear()
        marker_id = 1
        
        # Visualize completed polygons
        for i, polygon in enumerate(self.no_go_polygons):
            if len(polygon) >= 3:
                # Polygon area (filled)
                area_marker = Marker()
                area_marker.header.frame_id = self.global_frame_id
                area_marker.header.stamp = self.get_clock().now().to_msg()
                area_marker.ns = "no_go_polygons"
                area_marker.id = marker_id
                marker_id += 1
                area_marker.type = Marker.TRIANGLE_LIST
                area_marker.action = Marker.ADD
                area_marker.pose.orientation.w = 1.0
                area_marker.scale.x = 1.0
                area_marker.scale.y = 1.0
                area_marker.scale.z = 1.0
                area_marker.color.r = 1.0
                area_marker.color.g = 0.0
                area_marker.color.b = 0.0
                area_marker.color.a = 0.3
                
                # Simple triangulation for convex polygons
                for j in range(1, len(polygon) - 1):
                    # Triangle: polygon[0], polygon[j], polygon[j+1]
                    for k in [0, j, j+1]:
                        pt = Point()
                        pt.x = polygon[k][0]
                        pt.y = polygon[k][1]
                        pt.z = 0.01
                        area_marker.points.append(pt)
                
                marker_array.markers.append(area_marker)
                
                # Polygon outline
                outline_marker = Marker()
                outline_marker.header.frame_id = self.global_frame_id
                outline_marker.header.stamp = self.get_clock().now().to_msg()
                outline_marker.ns = "no_go_polygons"
                outline_marker.id = marker_id
                marker_id += 1
                outline_marker.type = Marker.LINE_STRIP
                outline_marker.action = Marker.ADD
                outline_marker.pose.orientation.w = 1.0
                outline_marker.scale.x = 0.05
                outline_marker.color.r = 0.8
                outline_marker.color.g = 0.0
                outline_marker.color.b = 0.0
                outline_marker.color.a = 1.0
                
                for px, py in polygon:
                    pt = Point()
                    pt.x = px
                    pt.y = py
                    pt.z = 0.02
                    outline_marker.points.append(pt)
                
                # Close the polygon
                pt = Point()
                pt.x = polygon[0][0]
                pt.y = polygon[0][1]
                pt.z = 0.02
                outline_marker.points.append(pt)
                
                marker_array.markers.append(outline_marker)
        
        # Visualize current polygon being built
        if self.current_polygon_points:
            current_marker = Marker()
            current_marker.header.frame_id = self.global_frame_id
            current_marker.header.stamp = self.get_clock().now().to_msg()
            current_marker.ns = "no_go_polygons"
            current_marker.id = marker_id
            current_marker.type = Marker.LINE_STRIP
            current_marker.action = Marker.ADD
            current_marker.pose.orientation.w = 1.0
            current_marker.scale.x = 0.03
            current_marker.color.r = 0.0
            current_marker.color.g = 1.0
            current_marker.color.b = 0.0
            current_marker.color.a = 1.0
            
            for px, py in self.current_polygon_points:
                pt = Point()
                pt.x = px
                pt.y = py
                pt.z = 0.03
                current_marker.points.append(pt)
            
            marker_array.markers.append(current_marker)
            
            # Show individual points of current polygon
            for j, (px, py) in enumerate(self.current_polygon_points):
                point_marker = Marker()
                point_marker.header.frame_id = self.global_frame_id
                point_marker.header.stamp = self.get_clock().now().to_msg()
                point_marker.ns = "no_go_polygons"
                point_marker.id = marker_id + j + 1
                point_marker.type = Marker.SPHERE
                point_marker.action = Marker.ADD
                point_marker.pose.position.x = px
                point_marker.pose.position.y = py
                point_marker.pose.position.z = 0.05
                point_marker.pose.orientation.w = 1.0
                point_marker.scale.x = 0.1
                point_marker.scale.y = 0.1
                point_marker.scale.z = 0.1
                point_marker.color.r = 0.0
                point_marker.color.g = 1.0
                point_marker.color.b = 1.0
                point_marker.color.a = 1.0
                marker_array.markers.append(point_marker)
        
        if marker_array.markers:
            self.marker_pub.publish(marker_array)

    def load_zones_from_file(self):
        try:
            if os.path.exists(self.zones_config_file_path):
                with open(self.zones_config_file_path, 'r') as f:
                    loaded_data = json.load(f)
                    # Handle both old format (list of points) and new format (list of polygons)
                    if loaded_data and isinstance(loaded_data[0], (list, tuple)):
                        if len(loaded_data[0]) == 2 and isinstance(loaded_data[0][0], (int, float)):
                            # Old format: list of [x, y] points, convert to single polygon
                            self.get_logger().info("Converting old point-based zones to single polygon.")
                            if len(loaded_data) >= self.min_polygon_points:
                                self.no_go_polygons = [[(float(pt[0]), float(pt[1])) for pt in loaded_data]]
                            else:
                                self.no_go_polygons = []
                        else:
                            # New format: list of polygons
                            self.no_go_polygons = []
                            for polygon_data in loaded_data:
                                if isinstance(polygon_data, list) and len(polygon_data) >= self.min_polygon_points:
                                    polygon = [(float(pt[0]), float(pt[1])) for pt in polygon_data if isinstance(pt, (list, tuple)) and len(pt) == 2]
                                    if len(polygon) >= self.min_polygon_points:
                                        self.no_go_polygons.append(polygon)
                    else:
                        self.no_go_polygons = []
                self.get_logger().info(f"Loaded {len(self.no_go_polygons)} no-go polygons from {self.zones_config_file_path}.")
            else:
                self.get_logger().info(f"No-go zone config file '{self.zones_config_file_path}' not found. Starting with empty list.")
        except Exception as e:
            self.get_logger().error(f"Error loading no-go zones from file '{self.zones_config_file_path}': {e}")

    def save_zones_to_file(self):
        try:
            with open(self.zones_config_file_path, 'w') as f:
                json.dump(self.no_go_polygons, f, indent=2)
            self.get_logger().info(f"Saved {len(self.no_go_polygons)} no-go polygons to {self.zones_config_file_path}.")
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