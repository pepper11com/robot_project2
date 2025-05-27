#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, Point, Pose
import numpy as np
import math
import heapq
import collections

from tf2_ros import Buffer, TransformListener
from rclpy.time import Time as rclpyTime
from rclpy.duration import Duration as rclpyDuration

LETHAL_OBSTACLE_COST_INTERNAL = 254
UNKNOWN_COST_INTERNAL = 253

class SimpleGlobalPlanner(Node):
    def __init__(self):
        super().__init__('simple_global_planner')

        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('path_publish_topic', '/global_path')
        self.declare_parameter('goal_topic', '/goal_pose')
        self.declare_parameter('robot_base_frame', 'base_link')
        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('inflation_radius_m', 0.5)
        self.declare_parameter('cost_lethal_threshold', 90)
        self.declare_parameter('cost_unknown_as_lethal', True)
        self.declare_parameter('cost_inflated_max', 200)
        self.declare_parameter('cost_neutral', 1)
        self.declare_parameter('cost_penalty_multiplier', 1.0)
        self.declare_parameter('enable_path_simplification', True)
        self.declare_parameter('simplification_obstacle_check_expansion_cells', 0)
        # NEW PARAMETER for simplification cost threshold
        self.declare_parameter('simplification_max_allowed_cost', 50) # Default: treat cells with cost >= 50 as obstacles for simplification

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

        self.map_data: OccupancyGrid | None = None
        self.costmap: np.ndarray | None = None
        self.map_info: OccupancyGrid.info | None = None

        self.map_sub = self.create_subscription(
            OccupancyGrid, self.map_topic, self._map_callback, 10)
        self.goal_sub = self.create_subscription(
            PoseStamped, self.goal_topic_name, self._goal_callback, 10)
        self.path_pub = self.create_publisher(Path, self.path_publish_topic, 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.get_logger().info(f"SimpleGlobalPlanner initialized. Global frame: '{self.global_frame}'.")
        self.get_logger().info(f" Inflation: Radius={self.inflation_radius_m}m, MaxCost={self.cost_inflated_max}, NeutralCost={self.cost_neutral}")
        self.get_logger().info(f" A* Penalty Multiplier: {self.cost_penalty_multiplier}")
        if self.enable_path_simplification:
            self.get_logger().info(f" Path Simplification ENABLED: CheckExpansion={self.simplification_obstacle_check_expansion_cells} cells, MaxAllowedCostForSimp={self.simplification_max_allowed_cost}")
        else:
            self.get_logger().info(" Path Simplification DISABLED.")

    # ... (_map_callback, _create_and_inflate_costmap, _inflate_obstacles are unchanged)
    def _map_callback(self, msg: OccupancyGrid): # No change
        self.get_logger().info(f"Received map: {msg.info.width}x{msg.info.height}, res: {msg.info.resolution}, frame: '{msg.header.frame_id}'")
        if msg.header.frame_id != self.global_frame:
            self.get_logger().warn(
                f"Received map in frame '{msg.header.frame_id}' but expected global frame '{self.global_frame}'. "
            )
        self.map_data = msg
        self.map_info = msg.info
        self._create_and_inflate_costmap()

    def _create_and_inflate_costmap(self): # No change
        if self.map_data is None or self.map_info is None or self.map_info.resolution == 0:
            self.get_logger().warn("Map data or info not available or resolution is zero, cannot create costmap.")
            return
        width = self.map_info.width
        height = self.map_info.height
        self.costmap = np.full((height, width), self.cost_neutral, dtype=np.uint8)
        raw_map_data = np.array(self.map_data.data).reshape((height, width))
        obstacle_indices = raw_map_data >= self.cost_lethal_threshold
        self.costmap[obstacle_indices] = LETHAL_OBSTACLE_COST_INTERNAL
        if self.cost_unknown_as_lethal:
            unknown_indices = raw_map_data == -1
            self.costmap[unknown_indices] = LETHAL_OBSTACLE_COST_INTERNAL
        self.get_logger().info("Base costmap created. Starting inflation...")
        self._inflate_obstacles()
        self.get_logger().info("Costmap inflation complete.")
        self.publish_inflated_costmap_for_debug()

    def _inflate_obstacles(self): # No change
        if self.costmap is None or self.map_info is None or self.map_info.resolution == 0:
            return
        height, width = self.costmap.shape
        inflation_radius_cells = int(math.ceil(self.inflation_radius_m / self.map_info.resolution))
        if inflation_radius_cells == 0:
             self.get_logger().info("Inflation radius in cells is 0, no inflation will occur.")
             return
        inflated_costmap_temp = np.copy(self.costmap)
        queue = collections.deque()
        # Mark visited based on initial lethal obstacles, so we don't re-queue them from neighbors
        visited = np.zeros_like(self.costmap, dtype=bool)


        for r in range(height):
            for c in range(width):
                if self.costmap[r, c] == LETHAL_OBSTACLE_COST_INTERNAL:
                    queue.append((r, c, 0))
                    visited[r,c] = True # Mark initial obstacles as visited for distance calculation

        while queue:
            curr_r, curr_c, dist_steps = queue.popleft()

            # Stop inflating beyond the radius for the current source obstacle
            if dist_steps >= inflation_radius_cells:
                continue

            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0: continue
                    nr, nc = curr_r + dr, curr_c + dc

                    if 0 <= nr < height and 0 <= nc < width:
                        # Only process if not already visited for this inflation wave *or*
                        # if it's not a primary lethal obstacle (which we don't want to overwrite)
                        if not visited[nr, nc] and inflated_costmap_temp[nr,nc] < LETHAL_OBSTACLE_COST_INTERNAL:
                            visited[nr, nc] = True # Mark as processed for this wave
                            new_dist_steps = dist_steps + 1 # Distance from *an* obstacle
                            
                            # Cost calculation based on distance from closest obstacle cell
                            # Using a more common quadratic decay (from Nav2 costmap_2d)
                            # cost = max_cost * (1 - (d / r))^2
                            # We want cost to go from cost_inflated_max down to cost_neutral
                            # factor = ( (inflation_radius_cells - new_dist_steps) / inflation_radius_cells )
                            # Let's use the formula from your original code, it was fine
                            factor = (1.0 - float(new_dist_steps) / inflation_radius_cells)**2
                            cost_val = int( (self.cost_inflated_max - self.cost_neutral) * factor + self.cost_neutral)
                            cost_val = max(self.cost_neutral, min(cost_val, self.cost_inflated_max))


                            # Update cell if new cost is higher (closer to an obstacle or higher base cost)
                            # Important: Ensure we don't overwrite a LETHAL_OBSTACLE_COST_INTERNAL with a lower inflated cost
                            if inflated_costmap_temp[nr, nc] < LETHAL_OBSTACLE_COST_INTERNAL:
                                inflated_costmap_temp[nr, nc] = max(inflated_costmap_temp[nr, nc], cost_val)
                            
                            if new_dist_steps < inflation_radius_cells:
                                queue.append((nr, nc, new_dist_steps))
        
        self.costmap = inflated_costmap_temp
    # ... (_world_to_map, _map_to_world, _get_robot_pose_in_global_frame are unchanged)
    def _world_to_map(self, world_x, world_y) -> tuple[int, int] | None: # No change
        if self.map_info is None: return None
        origin_x, origin_y, res = self.map_info.origin.position.x, self.map_info.origin.position.y, self.map_info.resolution
        if res == 0: return None
        map_c, map_r = int((world_x - origin_x) / res), int((world_y - origin_y) / res)
        if 0 <= map_c < self.map_info.width and 0 <= map_r < self.map_info.height: return map_r, map_c
        return None

    def _map_to_world(self, map_r, map_c) -> tuple[float, float] | None: # No change
        if self.map_info is None: return None
        origin_x, origin_y, res = self.map_info.origin.position.x, self.map_info.origin.position.y, self.map_info.resolution
        world_x, world_y = origin_x + (map_c + 0.5) * res, origin_y + (map_r + 0.5) * res
        return world_x, world_y

    def _get_robot_pose_in_global_frame(self) -> Pose | None: # No change
        try:
            now = rclpyTime()
            if not self.tf_buffer.can_transform(self.global_frame, self.robot_base_frame, now, rclpyDuration(seconds=0.5)):
                self.get_logger().warn(f"Cannot transform from '{self.global_frame}' to '{self.robot_base_frame}'. TF not available?", throttle_duration_sec=2.0)
                return None
            transform_stamped = self.tf_buffer.lookup_transform(self.global_frame, self.robot_base_frame, now, timeout=rclpyDuration(seconds=0.1))
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = transform_stamped.transform.translation.x, transform_stamped.transform.translation.y, transform_stamped.transform.translation.z
            return pose
        except Exception as e:
            self.get_logger().error(f"TF lookup from '{self.robot_base_frame}' to '{self.global_frame}' failed: {e}", throttle_duration_sec=1.0)
            return None
            
    # --- MODIFIED _is_cell_traversable ---
    def _is_cell_traversable(self, r: int, c: int, check_expansion: int = 0, cost_threshold: int = LETHAL_OBSTACLE_COST_INTERNAL) -> bool:
        """
        Checks if a cell and its optional expanded neighborhood are traversable
        below a given cost_threshold.
        """
        if self.costmap is None:
            self.get_logger().warn("_is_cell_traversable called with no costmap", throttle_duration_sec=5.0)
            return False # Should not happen if planning is active

        for dr_offset in range(-check_expansion, check_expansion + 1):
            for dc_offset in range(-check_expansion, check_expansion + 1):
                nr, nc = r + dr_offset, c + dc_offset
                if not (0 <= nr < self.costmap.shape[0] and 0 <= nc < self.costmap.shape[1]):
                    return False  # Expanded check goes out of bounds
                if self.costmap[nr, nc] >= cost_threshold: # Use the passed threshold
                    return False
        return True

    # --- MODIFIED _bresenham_line_check ---
    def _bresenham_line_check(self, p1_rc: tuple[int, int], p2_rc: tuple[int, int]) -> bool:
        """
        Checks if all cells on the line between p1_rc and p2_rc are traversable
        below the self.simplification_max_allowed_cost.
        """
        if self.costmap is None: return False # Should ideally not happen
        r1, c1 = p1_rc
        r2, c2 = p2_rc

        dr = abs(r2 - r1)
        dc = abs(c2 - c1)
        sr = 1 if r1 < r2 else -1 if r1 > r2 else 0
        sc = 1 if c1 < c2 else -1 if c1 > c2 else 0

        err = dr - dc
        curr_r, curr_c = r1, c1

        # Parameters for this check
        check_expansion = self.simplification_obstacle_check_expansion_cells
        cost_thresh_for_simplification = self.simplification_max_allowed_cost

        max_steps = dr + dc + 2 # Safety break
        steps_taken = 0

        while steps_taken < max_steps:
            if not self._is_cell_traversable(curr_r, curr_c, check_expansion, cost_thresh_for_simplification):
                return False # Blocked by inflated cost or obstacle within expansion

            if curr_r == r2 and curr_c == c2:
                break # Reached target

            e2 = 2 * err
            next_r, next_c = curr_r, curr_c # Store current before modification
            if e2 > -dc:
                err -= dc
                next_r += sr
            if e2 < dr:
                err += dr
                next_c += sc

            # If making a diagonal move, and expansion is 0 (meaning only center line checked),
            # explicitly check the two cells that form the 'corner' of the 1-cell-thick diagonal move.
            # This helps prevent cutting corners too sharply if expansion is minimal.
            if check_expansion == 0 and curr_r != next_r and curr_c != next_c: # Diagonal move
                if not self._is_cell_traversable(curr_r, next_c, 0, cost_thresh_for_simplification) or \
                   not self._is_cell_traversable(next_r, curr_c, 0, cost_thresh_for_simplification):
                    return False # Corner cell is blocked

            curr_r, curr_c = next_r, next_c # Update current position
            steps_taken += 1
            if steps_taken >= max_steps:
                self.get_logger().warn(f"Bresenham took too many steps from {p1_rc} to {p2_rc}. Assuming blocked.")
                return False
        return True

    # ... (_simplify_path_los is unchanged, it now uses the modified _bresenham_line_check)
    def _simplify_path_los(self, path_cells: list[tuple[int, int]]) -> list[tuple[int, int]]: # No change in this function itself
        if not path_cells or len(path_cells) <= 2:
            self.get_logger().debug(f"Path too short to simplify (len: {len(path_cells)})")
            return path_cells
        simplified_path = [path_cells[0]]
        current_path_idx = 0
        while current_path_idx < len(path_cells) - 1:
            farthest_reachable_original_idx = current_path_idx + 1
            for next_candidate_original_idx in range(current_path_idx + 2, len(path_cells)):
                if self._bresenham_line_check(path_cells[current_path_idx], path_cells[next_candidate_original_idx]):
                    farthest_reachable_original_idx = next_candidate_original_idx
                else:
                    break 
            simplified_path.append(path_cells[farthest_reachable_original_idx])
            current_path_idx = farthest_reachable_original_idx
        self.get_logger().info(f"Path simplified from {len(path_cells)} to {len(simplified_path)} points using LOS.")
        return simplified_path
        
    # ... (_goal_callback is unchanged, it calls _simplify_path_los)
    def _goal_callback(self, msg: PoseStamped): # No change
        if self.costmap is None or self.map_info is None or self.map_data is None:
            self.get_logger().warn("Costmap, map info, or map data not ready, cannot plan path yet.")
            return
        if msg.header.frame_id != self.global_frame:
            self.get_logger().warn(f"Goal received in frame '{msg.header.frame_id}', but planner operates in '{self.global_frame}'.")
        current_robot_pose = self._get_robot_pose_in_global_frame()
        if current_robot_pose is None:
            self.get_logger().error("Failed to get current robot pose. Cannot plan path.")
            return
        start_world_x, start_world_y = current_robot_pose.position.x, current_robot_pose.position.y
        goal_world_x, goal_world_y = msg.pose.position.x, msg.pose.position.y
        start_map_coords = self._world_to_map(start_world_x, start_world_y)
        goal_map_coords = self._world_to_map(goal_world_x, goal_world_y)
        if start_map_coords is None: self.get_logger().error(f"Robot start ({start_world_x:.2f}, {start_world_y:.2f}) outside map."); return
        if goal_map_coords is None: self.get_logger().error(f"Goal ({goal_world_x:.2f}, {goal_world_y:.2f}) outside map."); return
        start_r, start_c = start_map_coords
        goal_r, goal_c = goal_map_coords
        self.get_logger().info(f"Planning from robot at ({start_r},{start_c}) to goal at ({goal_r},{goal_c})")
        if self.costmap[start_r, start_c] >= LETHAL_OBSTACLE_COST_INTERNAL: self.get_logger().error(f"Robot start ({start_r},{start_c}) is in a lethal obstacle!"); return
        if self.costmap[goal_r, goal_c] >= LETHAL_OBSTACLE_COST_INTERNAL: self.get_logger().error(f"Goal ({goal_r},{goal_c}) is in a lethal obstacle!"); return
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

    # ... (_heuristic, _find_path_astar, _reconstruct_path, publish_inflated_costmap_for_debug are unchanged)
    def _heuristic(self, p1_rc, p2_rc): # No change
        return abs(p1_rc[0] - p2_rc[0]) + abs(p1_rc[1] - p2_rc[1])

    def _find_path_astar(self, start_rc, goal_rc): # No change
        if self.costmap is None: return []
        open_set = []; heapq.heappush(open_set, (0, start_rc))
        came_from = {}
        g_score = np.full(self.costmap.shape, float('inf')); g_score[start_rc] = 0
        f_score = np.full(self.costmap.shape, float('inf')); f_score[start_rc] = self._heuristic(start_rc, goal_rc)
        height, width = self.costmap.shape
        processed_nodes_count = 0
        while open_set:
            processed_nodes_count += 1
            if processed_nodes_count > height * width * 1.5: self.get_logger().warn("A* processed too many nodes, breaking loop."); return []
            _, current_rc = heapq.heappop(open_set)
            if current_rc == goal_rc: return self._reconstruct_path(came_from, current_rc)
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0: continue
                    neighbor_rc = (current_rc[0] + dr, current_rc[1] + dc)
                    nr, nc = neighbor_rc
                    if not (0 <= nr < height and 0 <= nc < width): continue
                    cell_cost_value = self.costmap[nr, nc]
                    if cell_cost_value >= LETHAL_OBSTACLE_COST_INTERNAL: continue
                    move_cost = math.sqrt(dr**2 + dc**2)
                    additional_penalty = max(0, (float(cell_cost_value) - self.cost_neutral) * self.cost_penalty_multiplier)
                    tentative_g_score = g_score[current_rc] + move_cost + additional_penalty
                    if tentative_g_score < g_score[neighbor_rc]:
                        came_from[neighbor_rc] = current_rc
                        g_score[neighbor_rc] = tentative_g_score
                        f_score[neighbor_rc] = tentative_g_score + self._heuristic(neighbor_rc, goal_rc)
                        heapq.heappush(open_set, (f_score[neighbor_rc], neighbor_rc))
        return []

    def _reconstruct_path(self, came_from, current_rc): # No change
        path = [current_rc]
        while current_rc in came_from: current_rc = came_from[current_rc]; path.append(current_rc)
        return path[::-1]

    def publish_inflated_costmap_for_debug(self): # No change
        if self.costmap is None or self.map_info is None or self.map_data is None: return
        debug_map_msg = OccupancyGrid(); debug_map_msg.header = self.map_data.header; debug_map_msg.info = self.map_info
        viz_data = np.zeros_like(self.costmap, dtype=np.int8)
        neutral_mask = (self.costmap <= self.cost_neutral); lethal_mask = (self.costmap >= LETHAL_OBSTACLE_COST_INTERNAL)
        inflated_mask = (~neutral_mask) & (~lethal_mask)
        viz_data[neutral_mask] = 0; viz_data[lethal_mask] = 100
        min_display_cost, max_display_cost = self.cost_neutral + 1, self.cost_inflated_max
        if max_display_cost > min_display_cost:
            costs_to_scale = self.costmap[inflated_mask]
            scaled_values = ((costs_to_scale.astype(float) - min_display_cost) / (max_display_cost - min_display_cost)) * (99.0 - 1.0) + 1.0
            viz_data[inflated_mask] = np.clip(scaled_values, 1, 99).astype(np.int8)
        else: viz_data[inflated_mask] = 50
        debug_map_msg.data = viz_data.flatten().tolist()
        if not hasattr(self, 'debug_costmap_pub'): self.debug_costmap_pub = self.create_publisher(OccupancyGrid, '/inflated_global_costmap', 10)
        self.debug_costmap_pub.publish(debug_map_msg)
        self.get_logger().info("Published debug inflated costmap.")

def main(args=None):
    rclpy.init(args=args)
    planner_node = SimpleGlobalPlanner()
    try:
        rclpy.spin(planner_node)
    except KeyboardInterrupt:
        planner_node.get_logger().info("Keyboard interrupt, shutting down.")
    except Exception as e:
        planner_node.get_logger().error(f"Unhandled exception: {e}", exc_info=True)
    finally:
        if rclpy.ok(): planner_node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()