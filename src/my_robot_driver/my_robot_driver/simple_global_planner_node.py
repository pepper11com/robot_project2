#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path, MapMetaData # Added MapMetaData for type hinting
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

TEMP_OBSTACLE_LETHAL_CORE_RADIUS_M = 0.25
TEMP_OBSTACLE_INFLATION_COST_AT_CORE_EDGE = 230


class SimpleGlobalPlanner(Node):
    def __init__(self):
        super().__init__("simple_global_planner_node")

        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("path_publish_topic", "/global_path")
        self.declare_parameter("goal_topic", "/goal_pose")
        self.declare_parameter("robot_base_frame", "base_link")
        self.declare_parameter("global_frame", "map")
        self.declare_parameter("inflation_radius_m", 0.3)
        self.declare_parameter("cost_lethal_threshold", 90)
        self.declare_parameter("cost_unknown_as_lethal", True)
        self.declare_parameter("cost_inflated_max", 220)
        self.declare_parameter("cost_neutral", 1)
        self.declare_parameter("cost_penalty_multiplier", 1.0)
        self.declare_parameter("enable_path_simplification", True)
        self.declare_parameter("simplification_obstacle_check_expansion_cells", 1)
        self.declare_parameter("simplification_max_allowed_cost", 100)
        self.declare_parameter("temp_avoidance_topic", "/temp_avoidance_points")
        self.declare_parameter("temp_avoidance_radius_m", 0.50)
        self.declare_parameter("temp_avoidance_point_lifetime_s", 7.0)
        self.declare_parameter("astar_turn_penalty_cost", 7.0)
        self.declare_parameter("simplification_temp_obstacle_clearance_radius_m", 0.25)
        self.declare_parameter("simplification_min_angle_change_deg", 5.0)
        self.declare_parameter("no_go_zones_topic", "/no_go_zones/costmap")

        self.map_topic = self.get_parameter("map_topic").value
        self.path_publish_topic = self.get_parameter("path_publish_topic").value
        self.goal_topic_name = self.get_parameter("goal_topic").value
        self.robot_base_frame = self.get_parameter("robot_base_frame").value
        self.global_frame = self.get_parameter("global_frame").value
        self.inflation_radius_m = self.get_parameter("inflation_radius_m").value
        self.cost_lethal_threshold = self.get_parameter("cost_lethal_threshold").value
        self.cost_unknown_as_lethal = self.get_parameter("cost_unknown_as_lethal").value
        self.cost_inflated_max = self.get_parameter("cost_inflated_max").value
        self.cost_neutral = self.get_parameter("cost_neutral").value
        self.cost_penalty_multiplier = self.get_parameter("cost_penalty_multiplier").value
        self.enable_path_simplification = self.get_parameter("enable_path_simplification").value
        self.simplification_obstacle_check_expansion_cells = self.get_parameter("simplification_obstacle_check_expansion_cells").value
        self.simplification_max_allowed_cost = self.get_parameter("simplification_max_allowed_cost").value
        self.temp_avoidance_topic_name = self.get_parameter("temp_avoidance_topic").value
        self.temp_avoidance_radius_m = self.get_parameter("temp_avoidance_radius_m").value
        self.temp_avoidance_point_lifetime_s = self.get_parameter("temp_avoidance_point_lifetime_s").value
        self.astar_turn_penalty_cost = self.get_parameter("astar_turn_penalty_cost").value
        self.simplification_temp_obstacle_clearance_radius_m = self.get_parameter("simplification_temp_obstacle_clearance_radius_m").value
        self.simplification_min_angle_change_deg = self.get_parameter("simplification_min_angle_change_deg").value
        self.no_go_zones_topic_name = self.get_parameter("no_go_zones_topic").value

        if (self.simplification_temp_obstacle_clearance_radius_m < TEMP_OBSTACLE_LETHAL_CORE_RADIUS_M):
            self.get_logger().warn(f"simplification_temp_obstacle_clearance_radius_m ({self.simplification_temp_obstacle_clearance_radius_m}) < TEMP_OBSTACLE_LETHAL_CORE_RADIUS_M ({TEMP_OBSTACLE_LETHAL_CORE_RADIUS_M}). Adjusting.")
            self.simplification_temp_obstacle_clearance_radius_m = TEMP_OBSTACLE_LETHAL_CORE_RADIUS_M
        if (self.simplification_temp_obstacle_clearance_radius_m > self.temp_avoidance_radius_m):
            self.get_logger().warn(f"simplification_temp_obstacle_clearance_radius_m ({self.simplification_temp_obstacle_clearance_radius_m}) > temp_avoidance_radius_m ({self.temp_avoidance_radius_m}). Adjusting.")
            self.simplification_temp_obstacle_clearance_radius_m = self.temp_avoidance_radius_m

        self.map_data: OccupancyGrid | None = None # Full map message
        self.base_costmap: np.ndarray | None = None
        self.planning_costmap: np.ndarray | None = None
        self.map_info: MapMetaData | None = None # Only MapMetaData

        self.no_go_zones_full_msg: OccupancyGrid | None = None # Store the full no_go_zones message
        self.no_go_costmap_data: np.ndarray | None = None

        map_qos = rclpy.qos.QoSProfile(
            depth=1,
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
            durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.map_sub = self.create_subscription(OccupancyGrid, self.map_topic, self._map_callback, map_qos)
        self.goal_sub = self.create_subscription(PoseStamped, self.goal_topic_name, self._goal_callback, 10)
        self.path_pub = self.create_publisher(Path, self.path_publish_topic, 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.temporary_avoidance_points: list[tuple[float, float, rclpyTime]] = []
        self.temp_avoidance_sub = self.create_subscription(
            Float32MultiArray, self.temp_avoidance_topic_name, self._temp_avoidance_callback, 10
        )
        self.no_go_sub = self.create_subscription(
            OccupancyGrid,
            self.no_go_zones_topic_name,
            self._no_go_costmap_callback,
            map_qos
        )

        self.get_logger().info(
            f"SimpleGlobalPlanner initialized. Simp. Temp Clearance: {self.simplification_temp_obstacle_clearance_radius_m}m. Simp. Min Angle: {self.simplification_min_angle_change_deg} deg."
        )
        self.get_logger().info(f"Subscribing to no-go zones on: {self.no_go_zones_topic_name}")


    def _no_go_costmap_callback(self, msg: OccupancyGrid):
        if self.map_info is None: # map_info comes from the main /map
            self.get_logger().warn("Received no-go costmap, but main map info is not yet available. Storing for later.")
            self.no_go_zones_full_msg = msg # Store full message
            try:
                self.no_go_costmap_data = np.array(msg.data, dtype=np.uint8).reshape((msg.info.height, msg.info.width))
            except ValueError as e:
                self.get_logger().error(f"Error reshaping no-go costmap data: {e}. Expected {msg.info.height*msg.info.width}, got {len(msg.data)}")
                self.no_go_zones_full_msg = None
                self.no_go_costmap_data = None
            return

        # Validate against the main map's geometry (self.map_info and self.map_data.header)
        if (msg.info.resolution != self.map_info.resolution or
            msg.info.width != self.map_info.width or
            msg.info.height != self.map_info.height or
            abs(msg.info.origin.position.x - self.map_info.origin.position.x) > 1e-4 or
            abs(msg.info.origin.position.y - self.map_info.origin.position.y) > 1e-4 or
            (self.map_data and msg.header.frame_id != self.map_data.header.frame_id)): # Use self.map_data.header
            self.get_logger().error(
                "No-go zone map_info or frame_id mismatch with main map. CANNOT apply no-go zones! "
                f"Frame: {msg.header.frame_id} vs {self.map_data.header.frame_id if self.map_data else 'N/A'}, "
                f"Res: {msg.info.resolution:.4f} vs {self.map_info.resolution:.4f}, "
                f"W: {msg.info.width} vs {self.map_info.width}, H: {msg.info.height} vs {self.map_info.height}, "
                f"OriginX: {msg.info.origin.position.x:.4f} vs {self.map_info.origin.position.x:.4f}, "
                f"OriginY: {msg.info.origin.position.y:.4f} vs {self.map_info.origin.position.y:.4f}"
            )
            self.no_go_zones_full_msg = None # Invalidate
            self.no_go_costmap_data = None
            return

        self.get_logger().info(f"Received and validated no-go zone costmap from '{self.no_go_zones_topic_name}'.")
        self.no_go_zones_full_msg = msg # Store full message
        try:
            self.no_go_costmap_data = np.array(msg.data, dtype=np.uint8).reshape((msg.info.height, msg.info.width))
            # Log statistics about no-go zones
            lethal_cells = np.sum(self.no_go_costmap_data >= 100)
            high_cost_cells = np.sum((self.no_go_costmap_data >= 95) & (self.no_go_costmap_data < 100))
            self.get_logger().info(f"No-go map: {lethal_cells} lethal cells, {high_cost_cells} high-cost cells")
        except ValueError as e:
            self.get_logger().error(f"Error reshaping no-go costmap data on validated message: {e}")
            self.no_go_zones_full_msg = None
            self.no_go_costmap_data = None
            return

    def _temp_avoidance_callback(self, msg: Float32MultiArray):
        if len(msg.data) == 2:
            x, y = msg.data[0], msg.data[1]
            self.get_logger().info(f"Received temporary avoidance point: ({x:.2f}, {y:.2f}) in map frame.")
            self.temporary_avoidance_points = [p for p in self.temporary_avoidance_points if not (math.isclose(p[0], x, abs_tol=0.1) and math.isclose(p[1], y, abs_tol=0.1))]
            self.temporary_avoidance_points.append((x, y, self.get_clock().now()))
        else:
            self.get_logger().warn(f"Invalid temp_avoidance_point msg. Expected 2 floats, got {len(msg.data)}.")

    def _cleanup_old_avoidance_points(self):
        now = self.get_clock().now()
        lifetime_duration = rclpyDuration(seconds=self.temp_avoidance_point_lifetime_s)
        initial_count = len(self.temporary_avoidance_points)
        self.temporary_avoidance_points = [p for p in self.temporary_avoidance_points if (now - p[2]) < lifetime_duration]
        if initial_count > len(self.temporary_avoidance_points):
            self.get_logger().debug(f"Cleaned up {initial_count - len(self.temporary_avoidance_points)} old avoidance points.")

    def _apply_temporary_avoidances(self):
        if self.base_costmap is None or self.map_info is None:
            self.planning_costmap = None
            return
        
        self.planning_costmap = np.copy(self.base_costmap)
        num_no_go_applied = 0

        # 1. Apply No-Go Zones with stricter thresholds
        if self.no_go_costmap_data is not None and self.no_go_zones_full_msg is not None and self.map_data is not None:
            # Validate geometry and frame ID before applying
            no_go_info = self.no_go_zones_full_msg.info
            if (no_go_info.resolution == self.map_info.resolution and
                no_go_info.width == self.map_info.width and
                no_go_info.height == self.map_info.height and
                abs(no_go_info.origin.position.x - self.map_info.origin.position.x) < 1e-4 and
                abs(no_go_info.origin.position.y - self.map_info.origin.position.y) < 1e-4 and
                self.no_go_zones_full_msg.header.frame_id == self.map_data.header.frame_id):

                # Use lower threshold for stricter no-go enforcement
                NO_GO_THRESHOLD_FROM_MSG = np.uint8(90)  # Lower threshold means more restrictive
                no_go_mask = self.no_go_costmap_data >= NO_GO_THRESHOLD_FROM_MSG
                self.planning_costmap[no_go_mask] = LETHAL_OBSTACLE_COST_INTERNAL
                num_no_go_applied = np.sum(no_go_mask)
                if num_no_go_applied > 0:
                    self.get_logger().info(f"Applied {num_no_go_applied} no-go zone cells to planning costmap (threshold: {NO_GO_THRESHOLD_FROM_MSG}).")
            else:
                self.get_logger().warn(
                    "Mismatch between current map_info/frame and stored no_go_map_info/frame during _apply_temporary_avoidances. "
                    "Skipping no-go zone application this cycle."
                )
        
        # 2. Apply Temporary Avoidance Points
        self._cleanup_old_avoidance_points()
        applied_temp_cnt = 0
        if self.temporary_avoidance_points and self.map_info.resolution > 0:
            outer_r_cells = int(self.temp_avoidance_radius_m / self.map_info.resolution)
            lethal_core_r_cells = min(
                int(TEMP_OBSTACLE_LETHAL_CORE_RADIUS_M / self.map_info.resolution),
                outer_r_cells,
            )
            h, w = self.planning_costmap.shape
            for ax, ay, _ in self.temporary_avoidance_points:
                mc = self._world_to_map(ax, ay)
                if mc:
                    cr, cc = mc
                    point_had_effect = False
                    for ro in range(-outer_r_cells, outer_r_cells + 1):
                        for co in range(-outer_r_cells, outer_r_cells + 1):
                            r, c = cr + ro, cc + co
                            if not (0 <= r < h and 0 <= c < w): continue
                            dist_cells = math.sqrt(ro**2 + co**2)
                            if dist_cells > outer_r_cells: continue
                            if self.planning_costmap[r, c] >= LETHAL_OBSTACLE_COST_INTERNAL: continue
                            tmp_pen = 0
                            if dist_cells <= lethal_core_r_cells:
                                tmp_pen = TEMP_OBSTACLE_PLANNING_CORE_COST
                            elif dist_cells <= outer_r_cells:
                                infl_band_w = outer_r_cells - lethal_core_r_cells
                                if infl_band_w < 1e-3:
                                    tmp_pen = TEMP_OBSTACLE_PLANNING_CORE_COST if lethal_core_r_cells > 0 else TEMP_OBSTACLE_INFLATION_COST_AT_CORE_EDGE
                                else:
                                    f = max(0.0, min(1.0, 1.0 - (dist_cells - lethal_core_r_cells) / infl_band_w))
                                    infl_v = int((TEMP_OBSTACLE_INFLATION_COST_AT_CORE_EDGE - self.cost_neutral) * f + self.cost_neutral)
                                    tmp_pen = max(self.cost_neutral, min(infl_v, TEMP_OBSTACLE_INFLATION_COST_AT_CORE_EDGE))
                            if tmp_pen > self.planning_costmap[r, c]:
                                self.planning_costmap[r, c] = np.uint8(tmp_pen)
                                point_had_effect = True
                    if point_had_effect: applied_temp_cnt += 1
            if applied_temp_cnt > 0:
                self.get_logger().info(f"Applied {applied_temp_cnt} temporary avoidance zones to planning costmap.")
        elif not self.temporary_avoidance_points:
             self.get_logger().debug("No temporary avoidance points to apply.")


    def _map_callback(self, msg: OccupancyGrid):
        self.get_logger().info(f"Map: {msg.info.width}x{msg.info.height}, res: {msg.info.resolution}, frame: '{msg.header.frame_id}'")
        if msg.header.frame_id != self.global_frame:
            self.get_logger().warn(f"Map frame '{msg.header.frame_id}' mismatch with configured global_frame '{self.global_frame}'.")
        
        map_changed_significantly = False
        if self.map_info is None or \
           msg.info.resolution != self.map_info.resolution or \
           msg.info.width != self.map_info.width or \
           msg.info.height != self.map_info.height or \
           abs(msg.info.origin.position.x - self.map_info.origin.position.x) > 1e-4 or \
           abs(msg.info.origin.position.y - self.map_info.origin.position.y) > 1e-4 :
            map_changed_significantly = True

        self.map_data = msg # Store full map message
        self.map_info = msg.info # Store just MapMetaData for convenience

        if map_changed_significantly or self.base_costmap is None:
            self.get_logger().info("Map geometry changed or first map. Re-creating base costmap.")
            self._create_and_inflate_costmap()
            if self.no_go_zones_full_msg: # If we have a stored no-go message
                self.get_logger().info("Map changed, re-validating stored no-go costmap.")
                self._no_go_costmap_callback(self.no_go_zones_full_msg) # Re-run validation

    def _create_and_inflate_costmap(self):
        if self.map_data is None or self.map_info is None or self.map_info.resolution == 0:
            self.get_logger().warn("Map data NA for _create_and_inflate_costmap.")
            return
        w, h = self.map_info.width, self.map_info.height
        self.base_costmap = np.full((h, w), self.cost_neutral, dtype=np.uint8)
        raw_map = np.array(self.map_data.data).reshape((h, w))
        self.base_costmap[raw_map >= self.cost_lethal_threshold] = LETHAL_OBSTACLE_COST_INTERNAL
        if self.cost_unknown_as_lethal:
            self.base_costmap[raw_map == -1] = UNKNOWN_COST_INTERNAL
        self.get_logger().info("Base costmap created. Inflating...")
        self._inflate_obstacles_on_basemap()
        self.get_logger().info("Base costmap inflation complete.")
        self.planning_costmap = np.copy(self.base_costmap)

    def _inflate_obstacles_on_basemap(self):
        if self.base_costmap is None or self.map_info is None or self.map_info.resolution == 0: return
        h, w = self.base_costmap.shape
        infl_r_cells = int(math.ceil(self.inflation_radius_m / self.map_info.resolution))
        if infl_r_cells == 0: return
        infl_copy = np.copy(self.base_costmap)
        q = collections.deque()
        visited = np.array(self.base_costmap >= LETHAL_OBSTACLE_COST_INTERNAL)
        for r_idx in range(h):
            for c_idx in range(w):
                if self.base_costmap[r_idx, c_idx] >= LETHAL_OBSTACLE_COST_INTERNAL:
                    q.append((r_idx, c_idx, 0))
        while q:
            cr, cc, d_steps = q.popleft()
            if d_steps >= infl_r_cells: continue
            for dr_ in [-1, 0, 1]:
                for dc_ in [-1, 0, 1]:
                    if dr_ == 0 and dc_ == 0: continue
                    nr, nc = cr + dr_, cc + dc_
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                        visited[nr, nc] = True
                        nd_steps = d_steps + 1
                        f_ = (1.0 - float(nd_steps) / infl_r_cells) ** 2
                        cv = int((self.cost_inflated_max - self.cost_neutral) * f_ + self.cost_neutral)
                        cv = max(self.cost_neutral, min(cv, self.cost_inflated_max))
                        if infl_copy[nr, nc] < LETHAL_OBSTACLE_COST_INTERNAL:
                            infl_copy[nr, nc] = max(infl_copy[nr, nc], np.uint8(cv))
                        if nd_steps < infl_r_cells:
                            q.append((nr, nc, nd_steps))
        self.base_costmap = infl_copy

    def _world_to_map(self, wx, wy) -> tuple[int, int] | None:
        if self.map_info is None: return None
        ox, oy, res = self.map_info.origin.position.x, self.map_info.origin.position.y, self.map_info.resolution
        if res == 0: return None
        mc, mr = int((wx - ox) / res), int((wy - oy) / res)
        if 0 <= mc < self.map_info.width and 0 <= mr < self.map_info.height: return mr, mc
        return None

    def _map_to_world(self, mr, mc) -> tuple[float, float] | None:
        if self.map_info is None: return None
        ox, oy, res = self.map_info.origin.position.x, self.map_info.origin.position.y, self.map_info.resolution
        return ox + (mc + 0.5) * res, oy + (mr + 0.5) * res

    def _get_robot_pose_in_global_frame(self) -> Pose | None:
        try:
            now = rclpyTime()
            if not self.tf_buffer.can_transform(self.global_frame, self.robot_base_frame, now, rclpyDuration(seconds=0.5)):
                self.get_logger().warn(f"Cannot transform from {self.robot_base_frame} to {self.global_frame}. TF not ready?", throttle_duration_sec=2.0)
                return None
            ts = self.tf_buffer.lookup_transform(self.global_frame, self.robot_base_frame, now, timeout=rclpyDuration(seconds=0.1))
            p = Pose()
            p.position.x, p.position.y, p.position.z = ts.transform.translation.x, ts.transform.translation.y, ts.transform.translation.z
            p.orientation = ts.transform.rotation
            return p
        except Exception as e:
            self.get_logger().error(f"TF lookup failed in _get_robot_pose_in_global_frame: {e}", throttle_duration_sec=2.0)
            return None

    def _is_cell_traversable(self, r, c, exp, thresh, cmap) -> bool:
        if cmap is None: return False
        for dro in range(-exp, exp + 1):
            for dco in range(-exp, exp + 1):
                nr, nc = r + dro, c + dco
                if not (0 <= nr < cmap.shape[0] and 0 <= nc < cmap.shape[1]): return False
                if cmap[nr, nc] >= thresh: return False
        return True

    def _is_cell_within_any_temp_avoidance_zone(self, r: int, c: int) -> bool:
        if not self.temporary_avoidance_points or self.map_info is None or self.map_info.resolution == 0: return False
        world_coords = self._map_to_world(r, c)
        if world_coords is None: return True
        world_x, world_y = world_coords
        radius_for_check = self.simplification_temp_obstacle_clearance_radius_m
        if self.simplification_obstacle_check_expansion_cells > 0 and self.map_info.resolution > 0:
            radius_for_check += self.simplification_obstacle_check_expansion_cells * self.map_info.resolution * 0.5
        final_check_radius_sq = radius_for_check**2
        for avoid_x_center, avoid_y_center, _ in self.temporary_avoidance_points:
            dist_sq = (world_x - avoid_x_center) ** 2 + (world_y - avoid_y_center) ** 2
            if dist_sq <= final_check_radius_sq: return True
        return False

    def _bresenham_line_check(self, p1_rc, p2_rc) -> bool:
        cmap_static = self.base_costmap
        if cmap_static is None: return False
        r1, c1 = p1_rc; r2, c2 = p2_rc
        dr_abs, dc_abs = abs(r2 - r1), abs(c2 - c1)
        sr, sc = np.sign(r2 - r1), np.sign(c2 - c1)
        err = dr_abs - dc_abs
        cr, cc = r1, c1
        max_s = dr_abs + dc_abs + 2
        s_taken = 0
        s_thresh = self.simplification_max_allowed_cost
        s_exp = self.simplification_obstacle_check_expansion_cells
        while s_taken < max_s:
            if not self._is_cell_traversable(cr, cc, s_exp, s_thresh, cmap_static): return False
            if self._is_cell_within_any_temp_avoidance_zone(cr, cc): return False
            if cr == r2 and cc == c2: return True
            e2 = 2 * err
            pr, pc = cr, cc
            if e2 > -dc_abs: err -= dc_abs; cr += sr
            if e2 < dr_abs: err += dr_abs; cc += sc
            if s_exp == 0 and cr != pr and cc != pc:
                if not self._is_cell_traversable(pr, cc, 0, s_thresh, cmap_static) or self._is_cell_within_any_temp_avoidance_zone(pr, cc): return False
                if not self._is_cell_traversable(cr, pc, 0, s_thresh, cmap_static) or self._is_cell_within_any_temp_avoidance_zone(cr, pc): return False
            s_taken += 1
        return False

    def _simplify_path_los(self, path_cells) -> list:
        if not path_cells or len(path_cells) <= 2: return path_cells
        simp_path = [path_cells[0]]
        curr_idx = 0
        while curr_idx < len(path_cells) - 1:
            far_idx = curr_idx + 1
            for next_cand_idx in range(curr_idx + 2, len(path_cells)):
                if self._bresenham_line_check(path_cells[curr_idx], path_cells[next_cand_idx]):
                    far_idx = next_cand_idx
                else: break
            simp_path.append(path_cells[far_idx])
            curr_idx = far_idx
        self.get_logger().info(f"LOS Simp: {len(path_cells)} -> {len(simp_path)} pts.")
        return simp_path

    def _normalize_angle_pi(self, angle: float) -> float:
        while angle > math.pi: angle -= 2.0 * math.pi
        while angle < -math.pi: angle += 2.0 * math.pi
        return angle

    def _simplify_path_by_angle(self, path_cells: list[tuple[int, int]], min_angle_change_rad: float) -> list[tuple[int, int]]:
        if len(path_cells) <= 2 or min_angle_change_rad <= 0: return path_cells
        simplified_path = [path_cells[0]]
        for i in range(1, len(path_cells) - 1):
            p_prev, p_curr, p_next = path_cells[i-1], path_cells[i], path_cells[i+1]
            v1_dc, v1_dr = p_curr[1] - p_prev[1], p_curr[0] - p_prev[0]
            v2_dc, v2_dr = p_next[1] - p_curr[1], p_next[0] - p_curr[0]
            if (v1_dr == 0 and v1_dc == 0) or (v2_dr == 0 and v2_dc == 0):
                if not (v1_dr == 0 and v1_dc == 0): simplified_path.append(p_curr)
                continue
            angle1 = math.atan2(v1_dr, v1_dc)
            angle2 = math.atan2(v2_dr, v2_dc)
            angle_diff = abs(self._normalize_angle_pi(angle2 - angle1))
            if angle_diff > min_angle_change_rad:
                if not simplified_path or simplified_path[-1] != p_curr: simplified_path.append(p_curr)
        if not simplified_path or simplified_path[-1] != path_cells[-1]:
            if len(path_cells) > 1: simplified_path.append(path_cells[-1])
        if len(simplified_path) > 1:
            final_cleaned_path = [simplified_path[0]]
            for i in range(1, len(simplified_path)):
                if simplified_path[i] != simplified_path[i-1]: final_cleaned_path.append(simplified_path[i])
            simplified_path = final_cleaned_path
        if len(simplified_path) < len(path_cells):
            self.get_logger().info(f"Angle Simp: Input {len(path_cells)} -> Output {len(simplified_path)} pts.")
        return simplified_path

    def _goal_callback(self, msg: PoseStamped):
        if self.base_costmap is None or self.map_info is None or self.map_data is None:
            self.get_logger().warn("Map not ready for goal callback.")
            return
        if msg.header.frame_id != self.global_frame:
            self.get_logger().warn(f"Goal frame '{msg.header.frame_id}' mismatch with planner's global_frame '{self.global_frame}'. Attempting to proceed, ensure TF is correct.")
        
        robot_pose = self._get_robot_pose_in_global_frame()
        if robot_pose is None:
            self.get_logger().error("No robot pose available for planning.")
            return
        
        self._apply_temporary_avoidances() # This now includes no-go zones
        
        if self.planning_costmap is None:
            self.get_logger().error("Planning costmap not available after applying avoidances.")
            return
        
        start_coords = self._world_to_map(robot_pose.position.x, robot_pose.position.y)
        goal_coords = self._world_to_map(msg.pose.position.x, msg.pose.position.y)
        
        if start_coords is None or goal_coords is None:
            self.get_logger().error(f"Start or Goal is outside map boundaries. Start: {start_coords}, Goal: {goal_coords}")
            self.publish_planning_costmap_for_debug()
            return
        
        start_r, start_c = start_coords
        goal_r, goal_c = goal_coords
        self.get_logger().info(f"Planning from ({start_r},{start_c}) to ({goal_r},{goal_c}).")
        
        if self.planning_costmap[start_r, start_c] >= LETHAL_OBSTACLE_COST_INTERNAL or \
           self.planning_costmap[goal_r, goal_c] >= LETHAL_OBSTACLE_COST_INTERNAL:
            self.get_logger().error(
                f"Start or Goal is in a lethal area of the planning_costmap. "
                f"Start cost: {self.planning_costmap[start_r, start_c]}, Goal cost: {self.planning_costmap[goal_r, goal_c]}"
            )
            self.publish_planning_costmap_for_debug()
            return

        path_cells = self._find_path_astar(start_coords, goal_coords)
        final_path_to_publish = []
        if path_cells:
            self.get_logger().info(f"Raw A* path: {len(path_cells)} points.")
            simplified_cells = path_cells
            if self.enable_path_simplification and len(path_cells) > 2:
                simplified_cells = self._simplify_path_los(simplified_cells)
                if len(simplified_cells) > 2:
                    min_angle_rad = math.radians(self.simplification_min_angle_change_deg)
                    if min_angle_rad > 1e-3:
                        simplified_cells = self._simplify_path_by_angle(simplified_cells, min_angle_rad)
            self.get_logger().info(f"Final simplified path: {len(simplified_cells)} points.")
            final_path_to_publish = simplified_cells
        else:
            self.get_logger().warn(f"No A* path found.")

        ros_path = Path()
        ros_path.header.stamp = self.get_clock().now().to_msg()
        ros_path.header.frame_id = self.map_data.header.frame_id # Path in map frame
        if final_path_to_publish:
            for r, c in final_path_to_publish:
                wc = self._map_to_world(r, c)
                if wc:
                    p = PoseStamped()
                    p.header = ros_path.header
                    p.pose.position.x, p.pose.position.y = wc[0], wc[1]
                    p.pose.orientation.w = 1.0
                    ros_path.poses.append(p)
        self.path_pub.publish(ros_path)
        self.publish_planning_costmap_for_debug()

    def _heuristic(self, p1, p2): return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def _find_path_astar(self, start_rc, goal_rc):
        if self.planning_costmap is None: return []
        open_set = []
        heapq.heappush(open_set, (self._heuristic(start_rc, goal_rc), 0, start_rc, None))
        came_from = {}
        g_score = np.full(self.planning_costmap.shape, float("inf"))
        g_score[start_rc] = 0
        h, w = self.planning_costmap.shape
        proc_nodes, max_proc = 0, h * w * 2.5
        while open_set:
            proc_nodes += 1
            if proc_nodes > max_proc: self.get_logger().warn(f"A* max nodes ({proc_nodes})"); return []
            f, g_tie, curr, parent = heapq.heappop(open_set)
            if g_tie > g_score[curr]: continue
            if curr == goal_rc: return self._reconstruct_path(came_from, curr)
            for dr_ in [-1, 0, 1]:
                for dc_ in [-1, 0, 1]:
                    if dr_ == 0 and dc_ == 0: continue
                    neighbor = (curr[0] + dr_, curr[1] + dc_)
                    nr, nc = neighbor
                    if not (0 <= nr < h and 0 <= nc < w): continue
                    ccv = self.planning_costmap[nr, nc]
                    if ccv >= LETHAL_OBSTACLE_COST_INTERNAL: continue
                    m_cost = math.sqrt(dr_**2 + dc_**2)
                    add_pen = (float(ccv) - self.cost_neutral) * self.cost_penalty_multiplier if ccv > self.cost_neutral else 0
                    turn_pen = 0
                    if parent is not None and self.astar_turn_penalty_cost > 1e-3:
                        vpcd_r, vpcd_c = curr[0] - parent[0], curr[1] - parent[1]
                        if not (np.sign(vpcd_r) == np.sign(dr_) and np.sign(vpcd_c) == np.sign(dc_)):
                            turn_pen = self.astar_turn_penalty_cost
                    tent_g = g_score[curr] + m_cost + add_pen + turn_pen
                    if tent_g < g_score[neighbor]:
                        came_from[neighbor] = curr
                        g_score[neighbor] = tent_g
                        new_f = tent_g + self._heuristic(neighbor, goal_rc)
                        heapq.heappush(open_set, (new_f, tent_g, neighbor, curr))
        return []

    def _reconstruct_path(self, came_from, current_rc):
        path = [current_rc]
        while current_rc in came_from: current_rc = came_from[current_rc]; path.append(current_rc)
        return path[::-1]

    def publish_planning_costmap_for_debug(self,):
        if self.planning_costmap is None or self.map_info is None or self.map_data is None: return
        debug_map_msg = OccupancyGrid()
        debug_map_msg.header = self.map_data.header # Use map_data's header
        debug_map_msg.header.stamp = self.get_clock().now().to_msg()
        debug_map_msg.info = self.map_info # Use map_info
        viz = np.zeros_like(self.planning_costmap, dtype=np.int8)
        slm = self.planning_costmap >= LETHAL_OBSTACLE_COST_INTERNAL
        tcm = (self.planning_costmap == TEMP_OBSTACLE_PLANNING_CORE_COST) & (~slm)
        tim = ((self.planning_costmap > self.cost_neutral) & (self.planning_costmap < TEMP_OBSTACLE_PLANNING_CORE_COST) & (self.planning_costmap >= self.cost_inflated_max + 1) & (~slm))
        sim = ((self.planning_costmap > self.cost_neutral) & (self.planning_costmap <= self.cost_inflated_max) & (~slm) & (~tcm) & (~tim))
        nm = self.planning_costmap <= self.cost_neutral
        viz[nm] = 0; viz[slm] = 100; viz[tcm] = 99
        if np.any(tim):
            c = self.planning_costmap[tim]; mn, mx = float(np.min(c)), float(np.max(c))
            s = np.full_like(c, 85, dtype=float) if mx - mn < 1e-3 else 70.0 + ((c.astype(float) - mn) / (mx - mn)) * (98.0 - 70.0)
            viz[tim] = np.clip(s, 70, 98).astype(np.int8)
        if np.any(sim):
            c = self.planning_costmap[sim]; mn, mx = float(np.min(c)), float(np.max(c))
            s = np.full_like(c, 35, dtype=float) if mx - mn < 1e-3 else 1.0 + ((c.astype(float) - mn) / (mx - mn)) * (69.0 - 1.0)
            viz[sim] = np.clip(s, 1, 69).astype(np.int8)
        debug_map_msg.data = viz.flatten().tolist()
        if not hasattr(self, "debug_planning_costmap_pub"):
            self.debug_planning_costmap_pub = self.create_publisher(
                OccupancyGrid, "/planning_costmap_debug",
                rclpy.qos.QoSProfile(depth=1, reliability=rclpy.qos.ReliabilityPolicy.RELIABLE, durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL)
            )
        self.debug_planning_costmap_pub.publish(debug_map_msg)
        self.get_logger().debug("Pub debug costmap.")

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = SimpleGlobalPlanner()
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node: node.get_logger().info("Ctrl-C, shutting down.")
    finally:
        if node and rclpy.ok(): node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == "__main__":
    main()