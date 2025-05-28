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
from std_msgs.msg import Float32MultiArray

LETHAL_OBSTACLE_COST_INTERNAL = 254
UNKNOWN_COST_INTERNAL = 253
TEMP_OBSTACLE_PLANNING_CORE_COST = LETHAL_OBSTACLE_COST_INTERNAL - 1

TEMP_OBSTACLE_LETHAL_CORE_RADIUS_M = 0.20
TEMP_OBSTACLE_INFLATION_COST_AT_CORE_EDGE = 230

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
        self.declare_parameter('cost_penalty_multiplier', 1.0) # Lowered
        self.declare_parameter('enable_path_simplification', True)
        self.declare_parameter('simplification_obstacle_check_expansion_cells', 1) # Or 0
        self.declare_parameter('simplification_max_allowed_cost', 75) # Potentially increased
        self.declare_parameter('temp_avoidance_topic', '/temp_avoidance_points')
        self.declare_parameter('temp_avoidance_radius_m', 0.40) # Outer boundary for temp inflation & simp check
        self.declare_parameter('temp_avoidance_point_lifetime_s', 7.0)
        self.declare_parameter('astar_turn_penalty_cost', 3.0) # New parameter for turn penalty

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
        self.astar_turn_penalty_cost = self.get_parameter('astar_turn_penalty_cost').value

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

        self.get_logger().info(f"SimpleGlobalPlanner initialized. A* Turn Penalty: {self.astar_turn_penalty_cost}.")

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

    def _apply_temporary_avoidances(self):
        if self.base_costmap is None or self.map_info is None:
            self.planning_costmap = None; return
        self._cleanup_old_avoidance_points()
        self.planning_costmap = np.copy(self.base_costmap)
        if not self.temporary_avoidance_points or self.map_info.resolution == 0: return

        outer_radius_cells = int(self.temp_avoidance_radius_m / self.map_info.resolution)
        lethal_core_radius_cells = min(int(TEMP_OBSTACLE_LETHAL_CORE_RADIUS_M / self.map_info.resolution), outer_radius_cells)
        height, width = self.planning_costmap.shape
        applied_count = 0
        for avoid_x, avoid_y, _ in self.temporary_avoidance_points:
            map_coords = self._world_to_map(avoid_x, avoid_y)
            if map_coords:
                center_r, center_c = map_coords; applied_count += 1
                for r_offset in range(-outer_radius_cells, outer_radius_cells + 1):
                    for c_offset in range(-outer_radius_cells, outer_radius_cells + 1):
                        r, c = center_r + r_offset, center_c + c_offset
                        if not (0 <= r < height and 0 <= c < width): continue
                        dist_cells = math.sqrt(r_offset**2 + c_offset**2)
                        if dist_cells > outer_radius_cells: continue
                        if self.planning_costmap[r, c] >= LETHAL_OBSTACLE_COST_INTERNAL: continue
                        
                        calculated_temp_penalty = 0
                        if dist_cells <= lethal_core_radius_cells:
                            calculated_temp_penalty = TEMP_OBSTACLE_PLANNING_CORE_COST
                        elif dist_cells <= outer_radius_cells:
                            inflation_band_width_cells = outer_radius_cells - lethal_core_radius_cells
                            if inflation_band_width_cells < 1e-3 :
                                calculated_temp_penalty = TEMP_OBSTACLE_PLANNING_CORE_COST
                            else:
                                factor = 1.0 - (dist_cells - lethal_core_radius_cells) / inflation_band_width_cells
                                factor = max(0.0, min(1.0, factor))
                                inflated_val = int((TEMP_OBSTACLE_INFLATION_COST_AT_CORE_EDGE - self.cost_neutral) * factor + self.cost_neutral)
                                calculated_temp_penalty = max(self.cost_neutral, min(inflated_val, TEMP_OBSTACLE_INFLATION_COST_AT_CORE_EDGE))
                        if calculated_temp_penalty > self.planning_costmap[r,c]:
                             self.planning_costmap[r, c] = np.uint8(calculated_temp_penalty)
        if applied_count > 0:
            self.get_logger().info(f"Applied {applied_count} INFLATED temporary avoidance zones.")

    def _map_callback(self, msg: OccupancyGrid):
        # ... (same as before)
        self.get_logger().info(f"Received map: {msg.info.width}x{msg.info.height}, res: {msg.info.resolution}, frame: '{msg.header.frame_id}'")
        if msg.header.frame_id != self.global_frame:
            self.get_logger().warn(f"Received map in frame '{msg.header.frame_id}' but expected global frame '{self.global_frame}'.")
        self.map_data = msg
        self.map_info = msg.info
        self._create_and_inflate_costmap()

    def _create_and_inflate_costmap(self):
        # ... (same as before)
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
        # ... (same as before)
        if self.base_costmap is None or self.map_info is None or self.map_info.resolution == 0: return
        height, width = self.base_costmap.shape
        inflation_radius_cells = int(math.ceil(self.inflation_radius_m / self.map_info.resolution))
        if inflation_radius_cells == 0: return
        inflated_copy = np.copy(self.base_costmap)
        queue = collections.deque()
        visited_for_inflation = np.array(self.base_costmap >= LETHAL_OBSTACLE_COST_INTERNAL)
        for r_idx in range(height):
            for c_idx in range(width):
                if self.base_costmap[r_idx, c_idx] >= LETHAL_OBSTACLE_COST_INTERNAL:
                    queue.append((r_idx, c_idx, 0))
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
                                inflated_copy[nr, nc] = max(inflated_copy[nr, nc], np.uint8(cost_val))
                            if new_dist_steps < inflation_radius_cells:
                                queue.append((nr, nc, new_dist_steps))
        self.base_costmap = inflated_copy

    def _world_to_map(self, world_x, world_y) -> tuple[int, int] | None:
        # ... (same as before)
        if self.map_info is None: return None
        origin_x, origin_y, res = self.map_info.origin.position.x, self.map_info.origin.position.y, self.map_info.resolution
        if res == 0: return None
        map_c, map_r = int((world_x - origin_x) / res), int((world_y - origin_y) / res)
        if 0 <= map_c < self.map_info.width and 0 <= map_r < self.map_info.height: return map_r, map_c
        return None

    def _map_to_world(self, map_r, map_c) -> tuple[float, float] | None:
        # ... (same as before)
        if self.map_info is None: return None
        origin_x, origin_y, res = self.map_info.origin.position.x, self.map_info.origin.position.y, self.map_info.resolution
        world_x, world_y = origin_x + (map_c + 0.5) * res, origin_y + (map_r + 0.5) * res
        return world_x, world_y

    def _get_robot_pose_in_global_frame(self) -> Pose | None:
        # ... (same as before)
        try:
            now = rclpyTime()
            if not self.tf_buffer.can_transform(self.global_frame, self.robot_base_frame, now, rclpyDuration(seconds=0.5)):
                self.get_logger().warn(f"TF not available: {self.global_frame} to {self.robot_base_frame}.", throttle_duration_sec=2.0); return None
            ts = self.tf_buffer.lookup_transform(self.global_frame, self.robot_base_frame, now, timeout=rclpyDuration(seconds=0.1))
            p = Pose(); p.position.x, p.position.y, p.position.z = ts.transform.translation.x, ts.transform.translation.y, ts.transform.translation.z
            p.orientation = ts.transform.rotation; return p
        except Exception as e: self.get_logger().error(f"TF lookup failed: {e}", throttle_duration_sec=1.0); return None


    def _is_cell_traversable(self, r: int, c: int, check_expansion: int = 0, cost_threshold: int = LETHAL_OBSTACLE_COST_INTERNAL, costmap_to_check: np.ndarray | None = None) -> bool:
        # ... (same as before)
        active_costmap = costmap_to_check
        if active_costmap is None: self.get_logger().warn("_is_cell_traversable no costmap", throttle_duration_sec=5.0); return False
        for dr_offset in range(-check_expansion, check_expansion + 1):
            for dc_offset in range(-check_expansion, check_expansion + 1):
                nr, nc = r + dr_offset, c + dc_offset
                if not (0 <= nr < active_costmap.shape[0] and 0 <= nc < active_costmap.shape[1]): return False
                if active_costmap[nr, nc] >= cost_threshold: return False
        return True

    def _is_cell_within_any_temp_avoidance_zone(self, r: int, c: int) -> bool:
        # ... (same as before, uses self.temp_avoidance_radius_m for hard simp boundary)
        if not self.temporary_avoidance_points or self.map_info is None or self.map_info.resolution == 0: return False
        world_coords = self._map_to_world(r, c)
        if world_coords is None: return True
        world_x, world_y = world_coords
        effective_check_radius_sq = (self.temp_avoidance_radius_m + self.simplification_obstacle_check_expansion_cells * self.map_info.resolution * 0.5)**2
        for avoid_x_center, avoid_y_center, _ in self.temporary_avoidance_points:
            if (world_x - avoid_x_center)**2 + (world_y - avoid_y_center)**2 <= effective_check_radius_sq: return True
        return False

    def _bresenham_line_check(self, p1_rc: tuple[int, int], p2_rc: tuple[int, int]) -> bool:
        # ... (same as before)
        costmap_for_static_checks = self.base_costmap
        if costmap_for_static_checks is None: self.get_logger().warn("Bresenham: base_costmap NA.", throttle_duration_sec=5.0); return False
        r1,c1=p1_rc; r2,c2=p2_rc; dr_abs,dc_abs=abs(r2-r1),abs(c2-c1); sr=1 if r1<r2 else -1 if r1>r2 else 0; sc=1 if c1<c2 else -1 if c1>c2 else 0
        err=dr_abs-dc_abs; curr_r,curr_c=r1,c1; max_steps=dr_abs+dc_abs+2; steps_taken=0
        cost_thresh=self.simplification_max_allowed_cost; check_exp=self.simplification_obstacle_check_expansion_cells
        while steps_taken < max_steps :
            if not self._is_cell_traversable(curr_r,curr_c,check_exp,cost_thresh,costmap_for_static_checks): return False
            if self._is_cell_within_any_temp_avoidance_zone(curr_r,curr_c): return False
            if curr_r==r2 and curr_c==c2: return True
            e2=2*err; prev_r,prev_c=curr_r,curr_c
            if e2 > -dc_abs: err-=dc_abs; curr_r+=sr
            if e2 < dr_abs: err+=dr_abs; curr_c+=sc
            if check_exp==0 and curr_r!=prev_r and curr_c!=prev_c:
                if not self._is_cell_traversable(prev_r,curr_c,0,cost_thresh,costmap_for_static_checks) or \
                   self._is_cell_within_any_temp_avoidance_zone(prev_r,curr_c): return False
                if not self._is_cell_traversable(curr_r,prev_c,0,cost_thresh,costmap_for_static_checks) or \
                   self._is_cell_within_any_temp_avoidance_zone(curr_r,prev_c): return False
            steps_taken+=1
        return False


    def _simplify_path_los(self, path_cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
        # ... (same as before)
        if not path_cells or len(path_cells) <= 2: return path_cells
        simplified_path = [path_cells[0]]; current_path_idx = 0
        while current_path_idx < len(path_cells) - 1:
            farthest_idx = current_path_idx + 1
            for next_candidate_idx in range(current_path_idx + 2, len(path_cells)):
                if self._bresenham_line_check(path_cells[current_path_idx], path_cells[next_candidate_idx]):
                    farthest_idx = next_candidate_idx
                else: break
            simplified_path.append(path_cells[farthest_idx]); current_path_idx = farthest_idx
        self.get_logger().info(f"Path simplified from {len(path_cells)} to {len(simplified_path)} points."); return simplified_path

    def _goal_callback(self, msg: PoseStamped):
        # ... (same as before, ensures _apply_temporary_avoidances is called)
        if self.base_costmap is None or self.map_info is None or self.map_data is None: self.get_logger().warn("Map not ready."); return
        if msg.header.frame_id != self.global_frame: self.get_logger().warn(f"Goal in wrong frame."); return
        robot_pose = self._get_robot_pose_in_global_frame()
        if robot_pose is None: self.get_logger().error("No robot pose."); return
        self._apply_temporary_avoidances()
        if self.planning_costmap is None: self.get_logger().error("No planning_costmap."); return
        start_coords = self._world_to_map(robot_pose.position.x, robot_pose.position.y)
        goal_coords = self._world_to_map(msg.pose.position.x, msg.pose.position.y)
        if start_coords is None or goal_coords is None: self.get_logger().error("Start/Goal outside map."); return
        start_r,start_c = start_coords; goal_r,goal_c = goal_coords
        self.get_logger().info(f"Planning from ({start_r},{start_c}) to ({goal_r},{goal_c}).")
        if self.planning_costmap[start_r,start_c] >= LETHAL_OBSTACLE_COST_INTERNAL or \
           self.planning_costmap[goal_r,goal_c] >= LETHAL_OBSTACLE_COST_INTERNAL:
            self.get_logger().error(f"Start/Goal in lethal obstacle. Start cost: {self.planning_costmap[start_r,start_c]}, Goal cost: {self.planning_costmap[goal_r,goal_c]}"); return
        
        path_cells = self._find_path_astar(start_coords, goal_coords)
        if path_cells:
            self.get_logger().info(f"Raw A* path: {len(path_cells)} points.")
            simplified_cells = self._simplify_path_los(path_cells) if self.enable_path_simplification and len(path_cells) > 2 else path_cells
            self.get_logger().info(f"Simplified path: {len(simplified_cells)} points.")
            ros_path = Path(); ros_path.header.stamp=self.get_clock().now().to_msg(); ros_path.header.frame_id=self.map_data.header.frame_id
            for r,c in simplified_cells:
                wc = self._map_to_world(r,c)
                if wc: p=PoseStamped(); p.header=ros_path.header; p.pose.position.x,p.pose.position.y=wc[0],wc[1]; p.pose.orientation.w=1.0; ros_path.poses.append(p)
            self.path_pub.publish(ros_path)
        else: self.get_logger().warn(f"No path found.")
        self.publish_planning_costmap_for_debug()


    def _heuristic(self, p1_rc, p2_rc): # Manhattan
        return abs(p1_rc[0] - p2_rc[0]) + abs(p1_rc[1] - p2_rc[1])

    # --- MODIFIED A* with Turn Penalty ---
    def _find_path_astar(self, start_rc: tuple[int,int], goal_rc: tuple[int,int]):
        if self.planning_costmap is None: return []
        
        # Heap stores: (f_score, g_score_for_tie_break, current_rc, parent_of_current_rc)
        # Tie-breaking with g_score can sometimes prefer shorter paths among those with equal f_score.
        open_set = []
        heapq.heappush(open_set, (self._heuristic(start_rc, goal_rc), 0, start_rc, None)) 
        
        came_from = {} 
        g_score = np.full(self.planning_costmap.shape, float('inf'))
        g_score[start_rc] = 0
        
        height, width = self.planning_costmap.shape
        processed_nodes_count = 0
        max_processed_nodes = height * width * 2.5 # Allow more processing for complex scenarios

        while open_set:
            processed_nodes_count += 1
            if processed_nodes_count > max_processed_nodes:
                self.get_logger().warn(f"A* max nodes ({processed_nodes_count}), breaking."); return []

            f_val, g_val_tie_break, current_rc, parent_of_current_rc = heapq.heappop(open_set)

            # Check if we've already found a better path to current_rc
            if g_val_tie_break > g_score[current_rc]: # Use actual g_score for this check
                 continue

            if current_rc == goal_rc:
                return self._reconstruct_path(came_from, current_rc)

            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0: continue
                    
                    neighbor_rc = (current_rc[0] + dr, current_rc[1] + dc)
                    nr, nc = neighbor_rc

                    if not (0 <= nr < height and 0 <= nc < width): continue
                    
                    cell_cost_value = self.planning_costmap[nr, nc]
                    if cell_cost_value >= LETHAL_OBSTACLE_COST_INTERNAL: continue
                                                                        
                    move_cost = math.sqrt(dr**2 + dc**2) 
                    
                    additional_penalty_from_cell_cost = 0
                    if cell_cost_value > self.cost_neutral:
                         additional_penalty_from_cell_cost = (float(cell_cost_value) - self.cost_neutral) * self.cost_penalty_multiplier
                    
                    turn_penalty = 0
                    if parent_of_current_rc is not None and self.astar_turn_penalty_cost > 1e-3:
                        vec_parent_current_dr = current_rc[0] - parent_of_current_rc[0]
                        vec_parent_current_dc = current_rc[1] - parent_of_current_rc[1]
                        # Normalize previous direction vector to compare with current dr, dc
                        # This simple check penalizes any deviation from a straight line grid move
                        norm_prev_dr = np.sign(vec_parent_current_dr)
                        norm_prev_dc = np.sign(vec_parent_current_dc)
                        norm_curr_dr = np.sign(dr)
                        norm_curr_dc = np.sign(dc)

                        if not (norm_prev_dr == norm_curr_dr and norm_prev_dc == norm_curr_dc):
                            # More sophisticated: check actual angle for "significant" turns
                            # For grid, any change in normalized direction components is a turn.
                            turn_penalty = self.astar_turn_penalty_cost
                    
                    tentative_g_score = g_score[current_rc] + move_cost + additional_penalty_from_cell_cost + turn_penalty

                    if tentative_g_score < g_score[neighbor_rc]:
                        came_from[neighbor_rc] = current_rc
                        g_score[neighbor_rc] = tentative_g_score
                        
                        new_f_score = tentative_g_score + self._heuristic(neighbor_rc, goal_rc)
                        heapq.heappush(open_set, (new_f_score, tentative_g_score, neighbor_rc, current_rc))
        return []
    # --- END OF MODIFIED A* ---

    def _reconstruct_path(self, came_from, current_rc):
        # ... (same as before)
        path = [current_rc];
        while current_rc in came_from: current_rc = came_from[current_rc]; path.append(current_rc)
        return path[::-1]

    def publish_planning_costmap_for_debug(self):
        # ... (same as before, visualization might need tweaking for new cost ranges)
        if self.planning_costmap is None or self.map_info is None or self.map_data is None: return
        debug_map_msg = OccupancyGrid(); debug_map_msg.header = self.map_data.header
        debug_map_msg.header.stamp = self.get_clock().now().to_msg(); debug_map_msg.info = self.map_info
        viz = np.zeros_like(self.planning_costmap, dtype=np.int8)
        sl_m=(self.planning_costmap>=LETHAL_OBSTACLE_COST_INTERNAL); tc_m=(self.planning_costmap==TEMP_OBSTACLE_PLANNING_CORE_COST)&(~sl_m)
        ti_m=(self.planning_costmap > self.cost_neutral)&(self.planning_costmap < TEMP_OBSTACLE_PLANNING_CORE_COST)&(self.planning_costmap >= self.cost_inflated_max +1 )&(~sl_m)
        si_m=(self.planning_costmap > self.cost_neutral)&(self.planning_costmap <= self.cost_inflated_max)&(~sl_m)&(~tc_m)&(~ti_m)
        n_m=(self.planning_costmap <= self.cost_neutral)
        viz[n_m]=0; viz[sl_m]=100; viz[tc_m]=99
        if np.any(ti_m): c=self.planning_costmap[ti_m]; mn,mx=float(np.min(c)),float(np.max(c)); s=np.full_like(c,85,dtype=float) if mx-mn<1e-3 else 70.+((c.astype(float)-mn)/(mx-mn))*(98.-70.); viz[ti_m]=np.clip(s,70,98).astype(np.int8)
        if np.any(si_m): c=self.planning_costmap[si_m]; mn,mx=float(np.min(c)),float(np.max(c)); s=np.full_like(c,35,dtype=float) if mx-mn<1e-3 else 1.+((c.astype(float)-mn)/(mx-mn))*(69.-1.); viz[si_m]=np.clip(s,1,69).astype(np.int8)
        debug_map_msg.data=viz.flatten().tolist()
        if not hasattr(self,'debug_pub'): self.debug_pub=self.create_publisher(OccupancyGrid,'/planning_costmap_debug',rclpy.qos.QoSProfile(depth=1,reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL))
        self.debug_pub.publish(debug_map_msg)
        self.get_logger().debug("Pub debug costmap.")


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = SimpleGlobalPlanner()
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node: node.get_logger().info("Ctrl-C, shutting down.")
    # except Exception as e: # Add back for general error catching if needed
    #     if node: node.get_logger().error(f"Unhandled exception: {e}", exc_info=True)
    #     else: print(f"Unhandled exception before node init: {e}")
    finally:
        if node and rclpy.ok(): node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()