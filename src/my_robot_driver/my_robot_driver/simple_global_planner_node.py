#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, Point, Pose
import numpy as np
import math
import heapq
import collections
from rclpy.time import Time as rclpyTime, Duration as rclpyDuration
from tf2_ros import Buffer, TransformListener
from std_msgs.msg import Float32MultiArray # For receiving temporary avoidance points

LETHAL_OBSTACLE_COST_INTERNAL = 254
UNKNOWN_COST_INTERNAL = 253
# Cost for the 'lethal core' of a temporary obstacle for A* planning.
# Should be high enough that A* strongly avoids it, but distinguishable from static lethal if needed for debug.
TEMP_OBSTACLE_PLANNING_CORE_COST = LETHAL_OBSTACLE_COST_INTERNAL - 1 # e.g., 253

# --- New Constants for Inflated Temporary Obstacles ---
# Radius of the 'solid' (highest cost) part of a temporary obstacle for A* planning.
# This should be roughly your robot's radius or slightly more.
TEMP_OBSTACLE_LETHAL_CORE_RADIUS_M = 0.20
# Cost value at the edge of the lethal core, from which inflation will decay outwards.
# Should be > self.cost_inflated_max (for static inflation) to make temp obstacles more repulsive.
TEMP_OBSTACLE_INFLATION_COST_AT_CORE_EDGE = 230
# ----------------------------------------------------

class SimpleGlobalPlanner(Node):
    def __init__(self):
        super().__init__('simple_global_planner_node')

        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('path_publish_topic', '/global_path')
        self.declare_parameter('goal_topic', '/goal_pose')
        self.declare_parameter('robot_base_frame', 'base_link')
        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('inflation_radius_m', 0.3)
        self.declare_parameter('cost_lethal_threshold', 90)
        self.declare_parameter('cost_unknown_as_lethal', True)
        self.declare_parameter('cost_inflated_max', 220)
        self.declare_parameter('cost_neutral', 1)
        self.declare_parameter('cost_penalty_multiplier', 2.0) # Adjusted default
        self.declare_parameter('enable_path_simplification', True)
        self.declare_parameter('simplification_obstacle_check_expansion_cells', 1)
        self.declare_parameter('simplification_max_allowed_cost', 50)
        self.declare_parameter('temp_avoidance_topic', '/temp_avoidance_points')
        # This 'temp_avoidance_radius_m' is now the OUTER boundary for temp obstacle inflation
        # AND the hard boundary for path simplification checks.
        self.declare_parameter('temp_avoidance_radius_m', 0.35) # Adjusted default
        self.declare_parameter('temp_avoidance_point_lifetime_s', 7.0)
        # Optional: make core radius and edge cost ROS params if more tuning needed
        # self.declare_parameter('temp_obstacle_lethal_core_radius_m', TEMP_OBSTACLE_LETHAL_CORE_RADIUS_M)
        # self.declare_parameter('temp_obstacle_inflation_cost_at_core_edge', TEMP_OBSTACLE_INFLATION_COST_AT_CORE_EDGE)


        self.map_topic = self.get_parameter('map_topic').value
        self.path_publish_topic = self.get_parameter('path_publish_topic').value
        self.goal_topic_name = self.get_parameter('goal_topic').value
        self.robot_base_frame = self.get_parameter('robot_base_frame').value
        self.global_frame = self.get_parameter('global_frame').value
        self.inflation_radius_m = self.get_parameter('inflation_radius_m').value
        self.cost_lethal_threshold = self.get_parameter('cost_lethal_threshold').value
        self.cost_unknown_as_lethal = self.get_parameter('cost_unknown_as_lethal').value
        self.cost_inflated_max = self.get_parameter('cost_inflated_max').value
        self.cost_neutral = self.get_parameter('cost_neutral').value
        self.cost_penalty_multiplier = self.get_parameter('cost_penalty_multiplier').value
        self.enable_path_simplification = self.get_parameter('enable_path_simplification').value
        self.simplification_obstacle_check_expansion_cells = self.get_parameter('simplification_obstacle_check_expansion_cells').value
        self.simplification_max_allowed_cost = self.get_parameter('simplification_max_allowed_cost').value
        self.temp_avoidance_topic_name = self.get_parameter('temp_avoidance_topic').value
        self.temp_avoidance_radius_m = self.get_parameter('temp_avoidance_radius_m').value
        self.temp_avoidance_point_lifetime_s = self.get_parameter('temp_avoidance_point_lifetime_s').value
        # self.temp_obstacle_lethal_core_radius_m_param = self.get_parameter('temp_obstacle_lethal_core_radius_m').value
        # self.temp_obstacle_inflation_cost_at_core_edge_param = self.get_parameter('temp_obstacle_inflation_cost_at_core_edge').value


        self.map_data: OccupancyGrid | None = None
        self.base_costmap: np.ndarray | None = None
        self.planning_costmap: np.ndarray | None = None
        self.map_info: OccupancyGrid.info | None = None

        map_qos = rclpy.qos.QoSProfile(depth=1, reliability=rclpy.qos.ReliabilityPolicy.RELIABLE, durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL)
        self.map_sub = self.create_subscription(OccupancyGrid, self.map_topic, self._map_callback, map_qos)
        self.goal_sub = self.create_subscription(PoseStamped, self.goal_topic_name, self._goal_callback, 10)
        self.path_pub = self.create_publisher(Path, self.path_publish_topic, 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.temporary_avoidance_points: list[tuple[float, float, rclpyTime]] = []
        self.temp_avoidance_sub = self.create_subscription(
            Float32MultiArray, self.temp_avoidance_topic_name, self._temp_avoidance_callback, 10)

        self.get_logger().info(f"SimpleGlobalPlanner initialized. Temp avoidance outer radius: {self.temp_avoidance_radius_m}m. Core radius for A*: {TEMP_OBSTACLE_LETHAL_CORE_RADIUS_M}m.")


    def _temp_avoidance_callback(self, msg: Float32MultiArray):
        if len(msg.data) == 2:
            x, y = msg.data[0], msg.data[1]
            self.get_logger().info(f"Received temporary avoidance point: ({x:.2f}, {y:.2f}) in map frame.")
            self.temporary_avoidance_points = [
                p for p in self.temporary_avoidance_points
                if not (math.isclose(p[0], x, abs_tol=0.1) and math.isclose(p[1], y, abs_tol=0.1))
            ]
            self.temporary_avoidance_points.append((x, y, self.get_clock().now()))
        else:
            self.get_logger().warn(f"Received invalid temp_avoidance_point message. Expected 2 floats, got {len(msg.data)}.")

    def _cleanup_old_avoidance_points(self):
        now = self.get_clock().now()
        lifetime_duration = rclpyDuration(seconds=self.temp_avoidance_point_lifetime_s)
        initial_count = len(self.temporary_avoidance_points)
        self.temporary_avoidance_points = [
            p for p in self.temporary_avoidance_points if (now - p[2]) < lifetime_duration
        ]
        if initial_count > len(self.temporary_avoidance_points):
            self.get_logger().debug(f"Cleaned up {initial_count - len(self.temporary_avoidance_points)} old avoidance points.")

    # --- MODIFIED METHOD ---
    def _apply_temporary_avoidances(self):
        if self.base_costmap is None or self.map_info is None:
            self.planning_costmap = None
            self.get_logger().warn("Base costmap or map info not available for applying temporary avoidances.")
            return

        self._cleanup_old_avoidance_points()
        self.planning_costmap = np.copy(self.base_costmap) # Start with static map + its inflation

        if not self.temporary_avoidance_points or self.map_info.resolution == 0:
            if self.temporary_avoidance_points and self.map_info.resolution == 0:
                 self.get_logger().debug("Map resolution is 0, cannot apply temporary avoidances.")
            return

        # Outer radius for any effect of the temporary obstacle (from ROS param)
        outer_radius_cells = int(self.temp_avoidance_radius_m / self.map_info.resolution)
        # Radius for the 'lethal' core of the temporary obstacle for A* (from global constant)
        # Use self.temp_obstacle_lethal_core_radius_m_param if made configurable
        lethal_core_radius_cells = int(TEMP_OBSTACLE_LETHAL_CORE_RADIUS_M / self.map_info.resolution)
        # Ensure lethal core is not larger than outer radius
        lethal_core_radius_cells = min(lethal_core_radius_cells, outer_radius_cells)


        height, width = self.planning_costmap.shape
        applied_count = 0

        for avoid_x, avoid_y, _ in self.temporary_avoidance_points:
            map_coords = self._world_to_map(avoid_x, avoid_y)
            if map_coords:
                center_r, center_c = map_coords
                applied_count += 1

                # Iterate in a square region around the temp obstacle center up to outer_radius
                for r_offset in range(-outer_radius_cells, outer_radius_cells + 1):
                    for c_offset in range(-outer_radius_cells, outer_radius_cells + 1):
                        r, c = center_r + r_offset, center_c + c_offset

                        if not (0 <= r < height and 0 <= c < width): # Cell out of bounds
                            continue

                        # Distance from current cell (r,c) to temp obstacle center (center_r, center_c)
                        dist_cells_sq = r_offset**2 + c_offset**2
                        dist_cells = math.sqrt(dist_cells_sq) # Distance in cells

                        # Skip if this cell is beyond the outer radius of influence for this temp point
                        if dist_cells > outer_radius_cells:
                            continue

                        # Start with the cost already on the planning_costmap (from base_costmap inflation)
                        new_cost_for_cell = self.planning_costmap[r, c]

                        # If cell is already marked as statically lethal, don't reduce its cost
                        if new_cost_for_cell >= LETHAL_OBSTACLE_COST_INTERNAL:
                            continue

                        calculated_temp_penalty = 0 # Penalty specifically from this temporary obstacle

                        if dist_cells <= lethal_core_radius_cells:
                            # Inside the 'lethal' core for A* planning
                            calculated_temp_penalty = TEMP_OBSTACLE_PLANNING_CORE_COST
                        elif dist_cells <= outer_radius_cells:
                            # Inside the inflation zone, outside the lethal core
                            inflation_band_width_cells = outer_radius_cells - lethal_core_radius_cells
                            if inflation_band_width_cells < 1e-3 : # Effectively no inflation band
                                if dist_cells <= outer_radius_cells: # Treat as core if band is non-existent
                                    calculated_temp_penalty = TEMP_OBSTACLE_PLANNING_CORE_COST
                            else:
                                normalized_dist_in_inflation_band = (dist_cells - lethal_core_radius_cells) / inflation_band_width_cells
                                # Linear decay: factor = 1.0 at core edge, 0.0 at outer_radius_cells
                                factor = 1.0 - normalized_dist_in_inflation_band
                                factor = max(0.0, min(1.0, factor)) # Clamp factor [0,1]

                                # Cost decays from TEMP_OBSTACLE_INFLATION_COST_AT_CORE_EDGE down to self.cost_neutral
                                # Use self.temp_obstacle_inflation_cost_at_core_edge_param if configurable
                                inflated_val = int((TEMP_OBSTACLE_INFLATION_COST_AT_CORE_EDGE - self.cost_neutral) * factor + self.cost_neutral)
                                calculated_temp_penalty = max(self.cost_neutral, min(inflated_val, TEMP_OBSTACLE_INFLATION_COST_AT_CORE_EDGE))
                        
                        # Update the planning_costmap cell if the temporary penalty is higher than its current cost
                        if calculated_temp_penalty > self.planning_costmap[r,c]:
                             self.planning_costmap[r, c] = np.uint8(calculated_temp_penalty)
        if applied_count > 0:
            self.get_logger().info(f"Applied {applied_count} INFLATED temporary avoidance zones to planning_costmap.")
    # --- END OF MODIFIED METHOD ---

    def _map_callback(self, msg: OccupancyGrid):
        self.get_logger().info(f"Received map: {msg.info.width}x{msg.info.height}, res: {msg.info.resolution}, frame: '{msg.header.frame_id}'")
        if msg.header.frame_id != self.global_frame:
            self.get_logger().warn(f"Received map in frame '{msg.header.frame_id}' but expected global frame '{self.global_frame}'.")
        self.map_data = msg
        self.map_info = msg.info
        self._create_and_inflate_costmap()

    def _create_and_inflate_costmap(self):
        if self.map_data is None or self.map_info is None or self.map_info.resolution == 0:
            self.get_logger().warn("Map data or info not available or resolution is zero, cannot create costmap.")
            return
        width = self.map_info.width
        height = self.map_info.height
        self.base_costmap = np.full((height, width), self.cost_neutral, dtype=np.uint8)
        raw_map_data = np.array(self.map_data.data).reshape((height, width))

        obstacle_indices = raw_map_data >= self.cost_lethal_threshold
        self.base_costmap[obstacle_indices] = LETHAL_OBSTACLE_COST_INTERNAL

        if self.cost_unknown_as_lethal:
            unknown_indices = raw_map_data == -1
            self.base_costmap[unknown_indices] = UNKNOWN_COST_INTERNAL

        self.get_logger().info("Base costmap created from static map. Starting inflation...")
        self._inflate_obstacles_on_basemap()
        self.get_logger().info("Base costmap inflation complete.")
        self.planning_costmap = np.copy(self.base_costmap)

    def _inflate_obstacles_on_basemap(self):
        if self.base_costmap is None or self.map_info is None or self.map_info.resolution == 0: return
        height, width = self.base_costmap.shape
        inflation_radius_cells = int(math.ceil(self.inflation_radius_m / self.map_info.resolution))
        if inflation_radius_cells == 0: return

        inflated_copy = np.copy(self.base_costmap)
        queue = collections.deque()
        visited_for_inflation = np.array(self.base_costmap >= LETHAL_OBSTACLE_COST_INTERNAL)

        for r in range(height):
            for c in range(width):
                if self.base_costmap[r, c] >= LETHAL_OBSTACLE_COST_INTERNAL:
                    queue.append((r, c, 0))

        while queue:
            curr_r, curr_c, dist_steps = queue.popleft()
            if dist_steps >= inflation_radius_cells: continue
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0: continue
                    nr, nc = curr_r + dr, curr_c + dc
                    if 0 <= nr < height and 0 <= nc < width:
                        if not visited_for_inflation[nr, nc]:
                            visited_for_inflation[nr, nc] = True
                            new_dist_steps = dist_steps + 1
                            factor = (1.0 - float(new_dist_steps) / inflation_radius_cells)**2
                            cost_val = int((self.cost_inflated_max - self.cost_neutral) * factor + self.cost_neutral)
                            cost_val = max(self.cost_neutral, min(cost_val, self.cost_inflated_max))
                            if inflated_copy[nr, nc] < LETHAL_OBSTACLE_COST_INTERNAL: # Check against original lethal
                                inflated_copy[nr, nc] = max(inflated_copy[nr, nc], np.uint8(cost_val))
                            if new_dist_steps < inflation_radius_cells:
                                queue.append((nr, nc, new_dist_steps))
        self.base_costmap = inflated_copy

    def _world_to_map(self, world_x, world_y) -> tuple[int, int] | None:
        if self.map_info is None: return None
        origin_x = self.map_info.origin.position.x
        origin_y = self.map_info.origin.position.y
        res = self.map_info.resolution
        if res == 0: return None
        map_c = int((world_x - origin_x) / res)
        map_r = int((world_y - origin_y) / res)
        if 0 <= map_c < self.map_info.width and 0 <= map_r < self.map_info.height:
            return map_r, map_c
        return None

    def _map_to_world(self, map_r, map_c) -> tuple[float, float] | None:
        if self.map_info is None: return None
        origin_x = self.map_info.origin.position.x
        origin_y = self.map_info.origin.position.y
        res = self.map_info.resolution
        world_x = origin_x + (map_c + 0.5) * res
        world_y = origin_y + (map_r + 0.5) * res
        return world_x, world_y

    def _get_robot_pose_in_global_frame(self) -> Pose | None:
        try:
            now = rclpyTime()
            if not self.tf_buffer.can_transform(self.global_frame, self.robot_base_frame, now, rclpyDuration(seconds=0.5)):
                self.get_logger().warn(f"Cannot transform from '{self.global_frame}' to '{self.robot_base_frame}'. TF not available?", throttle_duration_sec=2.0)
                return None
            transform_stamped = self.tf_buffer.lookup_transform(self.global_frame, self.robot_base_frame, now, timeout=rclpyDuration(seconds=0.1))
            pose = Pose()
            pose.position.x = transform_stamped.transform.translation.x
            pose.position.y = transform_stamped.transform.translation.y
            pose.position.z = transform_stamped.transform.translation.z
            pose.orientation = transform_stamped.transform.rotation
            return pose
        except Exception as e:
            self.get_logger().error(f"TF lookup from '{self.robot_base_frame}' to '{self.global_frame}' failed: {e}", throttle_duration_sec=1.0)
            return None

    def _is_cell_traversable(self, r: int, c: int, check_expansion: int = 0, cost_threshold: int = LETHAL_OBSTACLE_COST_INTERNAL, costmap_to_check: np.ndarray | None = None) -> bool:
        active_costmap = costmap_to_check
        if active_costmap is None:
            self.get_logger().warn("_is_cell_traversable called with no active_costmap", throttle_duration_sec=5.0)
            return False
        for dr_offset in range(-check_expansion, check_expansion + 1):
            for dc_offset in range(-check_expansion, check_expansion + 1):
                nr, nc = r + dr_offset, c + dc_offset
                if not (0 <= nr < active_costmap.shape[0] and 0 <= nc < active_costmap.shape[1]):
                    return False
                if active_costmap[nr, nc] >= cost_threshold:
                    return False
        return True

    def _is_cell_within_any_temp_avoidance_zone(self, r: int, c: int) -> bool:
        if not self.temporary_avoidance_points or self.map_info is None or self.map_info.resolution == 0:
            return False
        world_coords = self._map_to_world(r, c)
        if world_coords is None:
            return True
        world_x, world_y = world_coords
        
        # This radius (self.temp_avoidance_radius_m) is the hard boundary for simplification.
        # A* plans using an inflated zone *within* this.
        hard_boundary_radius_sq = self.temp_avoidance_radius_m ** 2
        
        # For simplification, consider a slight expansion if check_expansion_cells > 0
        # This is to ensure that if a cell center is clear, its expanded footprint for simplification check is also clear
        # of the hard boundary.
        effective_check_radius_sq = (self.temp_avoidance_radius_m + self.simplification_obstacle_check_expansion_cells * self.map_info.resolution * 0.5)**2


        for avoid_x_center, avoid_y_center, _ in self.temporary_avoidance_points:
            dist_sq = (world_x - avoid_x_center)**2 + (world_y - avoid_y_center)**2
            if dist_sq <= effective_check_radius_sq: # Check against the hard boundary for simplification
                return True
        return False

    def _bresenham_line_check(self, p1_rc: tuple[int, int], p2_rc: tuple[int, int]) -> bool:
        costmap_for_static_checks = self.base_costmap
        if costmap_for_static_checks is None:
            self.get_logger().warn("Bresenham: base_costmap not available.", throttle_duration_sec=5.0)
            return False
        r1, c1 = p1_rc; r2, c2 = p2_rc
        dr_abs, dc_abs = abs(r2 - r1), abs(c2 - c1)
        sr = 1 if r1 < r2 else -1 if r1 > r2 else 0
        sc = 1 if c1 < c2 else -1 if c1 > c2 else 0
        err = dr_abs - dc_abs
        curr_r, curr_c = r1, c1
        max_steps = dr_abs + dc_abs + 2
        steps_taken = 0
        cost_thresh_for_simplification_on_base = self.simplification_max_allowed_cost
        check_expansion = self.simplification_obstacle_check_expansion_cells

        while steps_taken < max_steps :
            if not self._is_cell_traversable(curr_r, curr_c, check_expansion,
                                             cost_thresh_for_simplification_on_base,
                                             costmap_for_static_checks):
                return False
            if self._is_cell_within_any_temp_avoidance_zone(curr_r, curr_c): # Check against hard boundary
                return False
            if curr_r == r2 and curr_c == c2:
                return True
            e2 = 2 * err
            prev_r, prev_c = curr_r, curr_c
            if e2 > -dc_abs: err -= dc_abs; curr_r += sr
            if e2 < dr_abs: err += dr_abs; curr_c += sc
            if check_expansion == 0 and curr_r != prev_r and curr_c != prev_c:
                if not self._is_cell_traversable(prev_r, curr_c, 0, cost_thresh_for_simplification_on_base, costmap_for_static_checks) or \
                   self._is_cell_within_any_temp_avoidance_zone(prev_r, curr_c):
                    return False
                if not self._is_cell_traversable(curr_r, prev_c, 0, cost_thresh_for_simplification_on_base, costmap_for_static_checks) or \
                   self._is_cell_within_any_temp_avoidance_zone(curr_r, prev_c):
                    return False
            steps_taken +=1
        return False

    def _simplify_path_los(self, path_cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if not path_cells or len(path_cells) <= 2: return path_cells
        simplified_path = [path_cells[0]]
        current_path_idx = 0
        while current_path_idx < len(path_cells) - 1:
            farthest_reachable_original_idx = current_path_idx + 1
            for next_candidate_original_idx in range(current_path_idx + 2, len(path_cells)):
                if self._bresenham_line_check(path_cells[current_path_idx], path_cells[next_candidate_original_idx]):
                    farthest_reachable_original_idx = next_candidate_original_idx
                else: break
            simplified_path.append(path_cells[farthest_reachable_original_idx])
            current_path_idx = farthest_reachable_original_idx
        self.get_logger().info(f"Path simplified from {len(path_cells)} to {len(simplified_path)} points using LOS.")
        return simplified_path

    def _goal_callback(self, msg: PoseStamped):
        if self.base_costmap is None or self.map_info is None or self.map_data is None:
            self.get_logger().warn("Base costmap, map info, or map data not ready, cannot plan path yet.")
            return
        if msg.header.frame_id != self.global_frame:
            self.get_logger().warn(f"Goal received in frame '{msg.header.frame_id}', but planner operates in '{self.global_frame}'.")
            return

        current_robot_pose = self._get_robot_pose_in_global_frame()
        if current_robot_pose is None:
            self.get_logger().error("Failed to get current robot pose. Cannot plan path.")
            return

        self._apply_temporary_avoidances() # This now creates inflated temporary zones
        if self.planning_costmap is None:
             self.get_logger().error("Planning costmap not available after applying temporary avoidances.")
             return

        start_world_x, start_world_y = current_robot_pose.position.x, current_robot_pose.position.y
        goal_world_x, goal_world_y = msg.pose.position.x, msg.pose.position.y
        start_map_coords = self._world_to_map(start_world_x, start_world_y)
        goal_map_coords = self._world_to_map(goal_world_x, goal_world_y)

        if start_map_coords is None: self.get_logger().error(f"Robot start ({start_world_x:.2f}, {start_world_y:.2f}) outside map."); return
        if goal_map_coords is None: self.get_logger().error(f"Goal ({goal_world_x:.2f}, {goal_world_y:.2f}) outside map."); return

        start_r, start_c = start_map_coords
        goal_r, goal_c = goal_map_coords
        self.get_logger().info(f"Planning from robot at map cell ({start_r},{start_c}) to goal cell ({goal_r},{goal_c}) using planning_costmap for A*.")

        if self.planning_costmap[start_r, start_c] >= LETHAL_OBSTACLE_COST_INTERNAL:
            self.get_logger().error(f"Robot start ({start_r},{start_c}) is in a lethal/core obstacle on planning_costmap! Cost: {self.planning_costmap[start_r, start_c]}"); return
        if self.planning_costmap[goal_r, goal_c] >= LETHAL_OBSTACLE_COST_INTERNAL:
            self.get_logger().error(f"Goal ({goal_r},{goal_c}) is in a lethal/core obstacle on planning_costmap! Cost: {self.planning_costmap[goal_r, goal_c]}"); return

        path_cells = self._find_path_astar((start_r, start_c), (goal_r, goal_c))

        if path_cells:
            self.get_logger().info(f"Raw A* path found with {len(path_cells)} points.")
            simplified_path_cells = self._simplify_path_los(path_cells) if self.enable_path_simplification and len(path_cells) > 2 else path_cells
            self.get_logger().info(f"Final path to publish has {len(simplified_path_cells)} points.")
            ros_path = Path()
            ros_path.header.stamp = self.get_clock().now().to_msg()
            ros_path.header.frame_id = self.map_data.header.frame_id
            for r_cell, c_cell in simplified_path_cells:
                world_coords = self._map_to_world(r_cell, c_cell)
                if world_coords:
                    pose = PoseStamped(); pose.header = ros_path.header
                    pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = world_coords[0], world_coords[1], 0.0
                    pose.pose.orientation.w = 1.0
                    ros_path.poses.append(pose)
            self.path_pub.publish(ros_path)
        else:
            self.get_logger().warn(f"No path found from ({start_r},{start_c}) to ({goal_r},{goal_c}).")
        self.publish_planning_costmap_for_debug()

    def _heuristic(self, p1_rc, p2_rc):
        return abs(p1_rc[0] - p2_rc[0]) + abs(p1_rc[1] - p2_rc[1])

    def _find_path_astar(self, start_rc, goal_rc):
        if self.planning_costmap is None: return []
        open_set = []; heapq.heappush(open_set, (0, start_rc))
        came_from = {}
        g_score = np.full(self.planning_costmap.shape, float('inf')); g_score[start_rc] = 0
        f_score = np.full(self.planning_costmap.shape, float('inf')); f_score[start_rc] = self._heuristic(start_rc, goal_rc)
        height, width = self.planning_costmap.shape
        processed_nodes_count = 0; max_processed_nodes = height * width * 1.5
        while open_set:
            processed_nodes_count += 1
            if processed_nodes_count > max_processed_nodes:
                self.get_logger().warn(f"A* processed too many nodes ({processed_nodes_count}), breaking."); return []
            current_f_score, current_rc = heapq.heappop(open_set)
            if current_f_score > f_score[current_rc]: continue
            if current_rc == goal_rc: return self._reconstruct_path(came_from, current_rc)
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0: continue
                    neighbor_rc = (current_rc[0] + dr, current_rc[1] + dc)
                    nr, nc = neighbor_rc
                    if not (0 <= nr < height and 0 <= nc < width): continue
                    
                    # Use cost from planning_costmap (includes static and temp inflated obstacles)
                    cell_cost_value = self.planning_costmap[nr, nc]
                    # LETHAL_OBSTACLE_COST_INTERNAL is the hard threshold for A*
                    if cell_cost_value >= LETHAL_OBSTACLE_COST_INTERNAL: continue 
                                                                        
                    move_cost = math.sqrt(dr**2 + dc**2)
                    additional_penalty = 0
                    # TEMP_OBSTACLE_PLANNING_CORE_COST is also a high cost that A* should penalize
                    if cell_cost_value > self.cost_neutral : # Penalty for any non-neutral cell
                         additional_penalty = (float(cell_cost_value) - self.cost_neutral) * self.cost_penalty_multiplier
                    
                    tentative_g_score = g_score[current_rc] + move_cost + additional_penalty
                    if tentative_g_score < g_score[neighbor_rc]:
                        came_from[neighbor_rc] = current_rc
                        g_score[neighbor_rc] = tentative_g_score
                        f_score[neighbor_rc] = tentative_g_score + self._heuristic(neighbor_rc, goal_rc)
                        heapq.heappush(open_set, (f_score[neighbor_rc], neighbor_rc))
        return []

    def _reconstruct_path(self, came_from, current_rc):
        path = [current_rc]
        while current_rc in came_from:
            current_rc = came_from[current_rc]
            path.append(current_rc)
        return path[::-1]

    def publish_planning_costmap_for_debug(self):
        if self.planning_costmap is None or self.map_info is None or self.map_data is None: return
        debug_map_msg = OccupancyGrid()
        debug_map_msg.header = self.map_data.header
        debug_map_msg.header.stamp = self.get_clock().now().to_msg()
        debug_map_msg.info = self.map_info
        viz_data = np.zeros_like(self.planning_costmap, dtype=np.int8)

        # Lethal from static map or unknown treated as lethal
        static_lethal_mask = (self.planning_costmap >= LETHAL_OBSTACLE_COST_INTERNAL)
        
        # Cells marked as temporary core by _apply_temporary_avoidances
        temp_core_mask = (self.planning_costmap == TEMP_OBSTACLE_PLANNING_CORE_COST) & (~static_lethal_mask)
        
        # Cells inflated by temporary obstacles (but not core and not static lethal)
        temp_inflated_mask = (self.planning_costmap > self.cost_neutral) & \
                             (self.planning_costmap < TEMP_OBSTACLE_PLANNING_CORE_COST) & \
                             (self.planning_costmap >= self.cost_inflated_max +1 ) & \
                             (~static_lethal_mask) # Assuming temp inflation starts above static inflation max
        
        # Cells inflated only by static map (not lethal, not temp core, not temp inflated)
        static_inflated_mask = (self.planning_costmap > self.cost_neutral) & \
                               (self.planning_costmap <= self.cost_inflated_max) & \
                               (~static_lethal_mask) & (~temp_core_mask) & (~temp_inflated_mask)

        neutral_mask = (self.planning_costmap <= self.cost_neutral)

        viz_data[neutral_mask] = 0
        viz_data[static_lethal_mask] = 100
        viz_data[temp_core_mask] = 99 # Temp lethal core visualization
        
        # Visualize temporary inflation (e.g., scale from 70-98)
        if np.any(temp_inflated_mask):
            costs = self.planning_costmap[temp_inflated_mask]
            min_c, max_c = float(np.min(costs)), float(np.max(costs))
            if max_c - min_c < 1e-3: scaled = np.full_like(costs, 85, dtype=float)
            else: scaled = 70.0 + ((costs.astype(float) - min_c) / (max_c - min_c)) * (98.0 - 70.0)
            viz_data[temp_inflated_mask] = np.clip(scaled, 70, 98).astype(np.int8)

        # Visualize static inflation (e.g., scale from 1-69)
        if np.any(static_inflated_mask):
            costs = self.planning_costmap[static_inflated_mask]
            min_c, max_c = float(np.min(costs)), float(np.max(costs))
            if max_c - min_c < 1e-3: scaled = np.full_like(costs, 35, dtype=float)
            else: scaled = 1.0 + ((costs.astype(float) - min_c) / (max_c - min_c)) * (69.0 - 1.0)
            viz_data[static_inflated_mask] = np.clip(scaled, 1, 69).astype(np.int8)
            
        debug_map_msg.data = viz_data.flatten().tolist()
        if not hasattr(self, 'debug_planning_costmap_pub'):
            self.debug_planning_costmap_pub = self.create_publisher(OccupancyGrid, '/planning_costmap_debug', 
                rclpy.qos.QoSProfile(depth=1, reliability=rclpy.qos.ReliabilityPolicy.RELIABLE, durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL))
        self.debug_planning_costmap_pub.publish(debug_map_msg)
        self.get_logger().debug("Published debug planning_costmap.")

def main(args=None):
    rclpy.init(args=args)
    planner_node = None
    try:
        planner_node = SimpleGlobalPlanner()
        rclpy.spin(planner_node)
    except KeyboardInterrupt:
        if planner_node: planner_node.get_logger().info("Keyboard interrupt, shutting down.")
    except Exception as e:
        if planner_node: planner_node.get_logger().error(f"Unhandled exception: {e}", exc_info=True)
        else: print(f"Unhandled exception before init: {e}")
    finally:
        if planner_node and rclpy.ok():
            if hasattr(planner_node, 'destroy_node') and callable(planner_node.destroy_node):
                planner_node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()