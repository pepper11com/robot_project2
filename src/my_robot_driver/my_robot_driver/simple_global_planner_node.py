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
TEMPORARY_AVOIDANCE_COST = 253 # VERY HIGH, but not lethal. Ensure cost_inflated_max < this.

class SimpleGlobalPlanner(Node):
    def __init__(self):
        super().__init__('simple_global_planner_node')

        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('path_publish_topic', '/global_path')
        self.declare_parameter('goal_topic', '/goal_pose')
        self.declare_parameter('robot_base_frame', 'base_link')
        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('inflation_radius_m', 0.3) # Global inflation
        self.declare_parameter('cost_lethal_threshold', 90) # From raw map
        self.declare_parameter('cost_unknown_as_lethal', True)
        self.declare_parameter('cost_inflated_max', 220) # Max for regular inflation
        self.declare_parameter('cost_neutral', 1)
        self.declare_parameter('cost_penalty_multiplier', 5.0) # Higher penalty
        self.declare_parameter('enable_path_simplification', True)
        self.declare_parameter('simplification_obstacle_check_expansion_cells', 1)
        self.declare_parameter('simplification_max_allowed_cost', 50)
        self.declare_parameter('temp_avoidance_topic', '/temp_avoidance_points')
        self.declare_parameter('temp_avoidance_radius_m', 0.75)
        self.declare_parameter('temp_avoidance_point_lifetime_s', 7.0)


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

        self.get_logger().info(f"SimpleGlobalPlanner initialized. Listening for map on '{self.map_topic}', goals on '{self.goal_topic_name}'.")
        self.get_logger().info(f"Temporary avoidance enabled on '{self.temp_avoidance_topic_name}', radius {self.temp_avoidance_radius_m}m, lifetime {self.temp_avoidance_point_lifetime_s}s.")


    def _temp_avoidance_callback(self, msg: Float32MultiArray):
        if len(msg.data) == 2:
            x, y = msg.data[0], msg.data[1]
            self.get_logger().info(f"Received temporary avoidance point: ({x:.2f}, {y:.2f}) in map frame.")
            self.temporary_avoidance_points = [p for p in self.temporary_avoidance_points if not (math.isclose(p[0], x, abs_tol=0.1) and math.isclose(p[1], y, abs_tol=0.1))] # Remove near duplicates
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


    def _apply_temporary_avoidances(self):
        if self.base_costmap is None or self.map_info is None:
            self.planning_costmap = None
            self.get_logger().warn("Base costmap or map info not available for applying temporary avoidances.")
            return
        
        self._cleanup_old_avoidance_points()
        self.planning_costmap = np.copy(self.base_costmap) 

        if not self.temporary_avoidance_points or self.map_info.resolution == 0:
            if self.temporary_avoidance_points:
                 self.get_logger().debug("Map resolution is 0, cannot apply temporary avoidances.")
            return

        radius_cells = int(self.temp_avoidance_radius_m / self.map_info.resolution)
        height, width = self.planning_costmap.shape

        applied_count = 0
        for avoid_x, avoid_y, _ in self.temporary_avoidance_points:
            map_coords = self._world_to_map(avoid_x, avoid_y)
            if map_coords:
                center_r, center_c = map_coords
                for r_offset in range(-radius_cells, radius_cells + 1):
                    for c_offset in range(-radius_cells, radius_cells + 1):
                        # Check if within circular radius
                        if r_offset**2 + c_offset**2 <= radius_cells**2:
                            r, c = center_r + r_offset, center_c + c_offset
                            if 0 <= r < height and 0 <= c < width:
                                # Only increase cost, don't overwrite lethal obstacles or lower existing high costs
                                if self.planning_costmap[r,c] < TEMPORARY_AVOIDANCE_COST and \
                                   self.planning_costmap[r,c] < LETHAL_OBSTACLE_COST_INTERNAL: # Don't mark lethal as less
                                     self.planning_costmap[r, c] = TEMPORARY_AVOIDANCE_COST
                applied_count +=1
        if applied_count > 0:
            self.get_logger().info(f"Applied {applied_count} temporary avoidance zones to planning_costmap.")


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
            self.base_costmap[unknown_indices] = LETHAL_OBSTACLE_COST_INTERNAL
        self.get_logger().info("Base costmap created from static map. Starting inflation...")
        self._inflate_obstacles_on_basemap()
        self.get_logger().info("Base costmap inflation complete.")

    def _inflate_obstacles_on_basemap(self):
        if self.base_costmap is None or self.map_info is None or self.map_info.resolution == 0: return
        height, width = self.base_costmap.shape
        inflation_radius_cells = int(math.ceil(self.inflation_radius_m / self.map_info.resolution))
        if inflation_radius_cells == 0: return

        inflated_copy = np.copy(self.base_costmap)
        queue = collections.deque()
        visited_for_inflation = np.array(self.base_costmap == LETHAL_OBSTACLE_COST_INTERNAL)

        for r in range(height):
            for c in range(width):
                if self.base_costmap[r, c] == LETHAL_OBSTACLE_COST_INTERNAL:
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
                            if inflated_copy[nr, nc] < LETHAL_OBSTACLE_COST_INTERNAL:
                                inflated_copy[nr, nc] = max(inflated_copy[nr, nc], cost_val)
                            if new_dist_steps < inflation_radius_cells:
                                queue.append((nr, nc, new_dist_steps))
        self.base_costmap = inflated_copy

    # ... (_world_to_map, _map_to_world, _get_robot_pose_in_global_frame as before) ...
    # ... (_is_cell_traversable, _bresenham_line_check, _simplify_path_los as before, ensure they use planning_costmap where appropriate for A*) ...
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
            return pose
        except Exception as e:
            self.get_logger().error(f"TF lookup from '{self.robot_base_frame}' to '{self.global_frame}' failed: {e}", throttle_duration_sec=1.0)
            return None

    def _is_cell_traversable(self, r: int, c: int, check_expansion: int = 0, cost_threshold: int = LETHAL_OBSTACLE_COST_INTERNAL, costmap_to_check: np.ndarray | None = None) -> bool:
        active_costmap = costmap_to_check if costmap_to_check is not None else self.planning_costmap
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

    def _bresenham_line_check(self, p1_rc: tuple[int, int], p2_rc: tuple[int, int]) -> bool:
        active_costmap_for_los = self.planning_costmap # Simplification can use the costlier map
        if active_costmap_for_los is None: return False
        r1, c1 = p1_rc; r2, c2 = p2_rc
        dr, dc = abs(r2 - r1), abs(c2 - c1)
        sr = 1 if r1 < r2 else -1 if r1 > r2 else 0
        sc = 1 if c1 < c2 else -1 if c1 > c2 else 0
        err = dr - dc
        curr_r, curr_c = r1, c1
        max_steps, steps_taken = dr + dc + 2, 0
        cost_thresh_for_simplification = self.simplification_max_allowed_cost
        check_expansion = self.simplification_obstacle_check_expansion_cells
        while steps_taken < max_steps:
            if not self._is_cell_traversable(curr_r, curr_c, check_expansion, cost_thresh_for_simplification, active_costmap_for_los): return False
            if curr_r == r2 and curr_c == c2: break
            e2 = 2 * err
            next_r, next_c = curr_r, curr_c
            if e2 > -dc: err -= dc; next_r += sr
            if e2 < dr: err += dr; next_c += sc
            if check_expansion == 0 and curr_r != next_r and curr_c != next_c:
                if not self._is_cell_traversable(curr_r, next_c, 0, cost_thresh_for_simplification, active_costmap_for_los) or \
                   not self._is_cell_traversable(next_r, curr_c, 0, cost_thresh_for_simplification, active_costmap_for_los): return False
            curr_r, curr_c = next_r, next_c
            steps_taken +=1
            if steps_taken >= max_steps: return False
        return True

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
        
        current_robot_pose = self._get_robot_pose_in_global_frame()
        if current_robot_pose is None:
            self.get_logger().error("Failed to get current robot pose. Cannot plan path.")
            return

        self._apply_temporary_avoidances()
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
        self.get_logger().info(f"Planning from robot at map cell ({start_r},{start_c}) to goal cell ({goal_r},{goal_c}) using planning_costmap.")

        if self.planning_costmap[start_r, start_c] >= LETHAL_OBSTACLE_COST_INTERNAL: 
            self.get_logger().error(f"Robot start ({start_r},{start_c}) is in a lethal obstacle on planning_costmap!"); return
        if self.planning_costmap[goal_r, goal_c] >= LETHAL_OBSTACLE_COST_INTERNAL: 
            self.get_logger().error(f"Goal ({goal_r},{goal_c}) is in a lethal obstacle on planning_costmap!"); return

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
                    pose = PoseStamped()
                    pose.header = ros_path.header 
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
        processed_nodes_count = 0
        while open_set:
            processed_nodes_count += 1
            if processed_nodes_count > height * width * 2: 
                self.get_logger().warn("A* processed too many nodes, breaking loop."); return []
            _, current_rc = heapq.heappop(open_set)
            if current_rc == goal_rc: return self._reconstruct_path(came_from, current_rc)
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0: continue
                    neighbor_rc = (current_rc[0] + dr, current_rc[1] + dc)
                    nr, nc = neighbor_rc
                    if not (0 <= nr < height and 0 <= nc < width): continue
                    cell_cost_value = self.planning_costmap[nr, nc]
                    if cell_cost_value >= LETHAL_OBSTACLE_COST_INTERNAL: continue
                    move_cost = math.sqrt(dr**2 + dc**2)
                    additional_penalty = 0
                    if cell_cost_value > self.cost_neutral:
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
        neutral_mask = (self.planning_costmap <= self.cost_neutral)
        lethal_mask = (self.planning_costmap >= LETHAL_OBSTACLE_COST_INTERNAL)
        # Visualize temporary avoidance zones with a distinct value (e.g., 99)
        temp_avoid_mask = (self.planning_costmap == TEMPORARY_AVOIDANCE_COST) & (~lethal_mask)
        # Regular inflation: not neutral, not lethal, not temporary avoidance
        inflated_mask = (~neutral_mask) & (~lethal_mask) & (~temp_avoid_mask)

        viz_data[neutral_mask] = 0
        viz_data[lethal_mask] = 100
        viz_data[temp_avoid_mask] = 99 # Make temp zones highly visible
        
        min_display_cost = self.cost_neutral + 1
        # Scale regular inflation up to a value just below temp_avoid_mask's visualization value
        max_display_cost_regular_inflation = min(self.cost_inflated_max, 98) 

        if max_display_cost_regular_inflation > min_display_cost:
            costs_to_scale = self.planning_costmap[inflated_mask]
            costs_to_scale_clipped = np.clip(costs_to_scale, min_display_cost, max_display_cost_regular_inflation)
            
            # Avoid division by zero if max and min are the same after clipping
            range_val = max_display_cost_regular_inflation - min_display_cost
            if range_val < 1e-3 : range_val = 1e-3 # prevent div by zero

            scaled_values = ((costs_to_scale_clipped.astype(float) - min_display_cost) / range_val) * (98.0 - 1.0) + 1.0
            viz_data[inflated_mask] = np.clip(scaled_values, 1, 98).astype(np.int8)
        else: 
            viz_data[inflated_mask] = 50 # Default for small range
            
        debug_map_msg.data = viz_data.flatten().tolist()
        if not hasattr(self, 'debug_planning_costmap_pub'):
            self.debug_planning_costmap_pub = self.create_publisher(OccupancyGrid, '/planning_costmap_debug', 
                rclpy.qos.QoSProfile(depth=1, reliability=rclpy.qos.ReliabilityPolicy.RELIABLE, durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL)) # Latch for RViz
        self.debug_planning_costmap_pub.publish(debug_map_msg)
        self.get_logger().debug("Published debug planning_costmap.")

def main(args=None):
    rclpy.init(args=args)
    planner_node = SimpleGlobalPlanner()
    try:
        rclpy.spin(planner_node)
    except KeyboardInterrupt:
        planner_node.get_logger().info("Keyboard interrupt, shutting down SimpleGlobalPlanner.")
    except Exception as e:
        planner_node.get_logger().error(f"Unhandled exception in SimpleGlobalPlanner: {e}", exc_info=True)
    finally:
        if rclpy.ok():
            if hasattr(planner_node, 'destroy_node') and callable(planner_node.destroy_node):
                planner_node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()