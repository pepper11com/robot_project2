#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration as rclpyDuration
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.time import Time as rclpyTime

from geometry_msgs.msg import Twist, PoseStamped, Point
from nav_msgs.msg import Path, OccupancyGrid
from std_msgs.msg import Float32MultiArray 
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs 
from tf_transformations import euler_from_quaternion
import numpy as np

CLIP = lambda v, lo, hi: max(lo, min(hi, v))

class RobotState:
    IDLE = 0
    ALIGNING = 1
    PAUSING_AFTER_ALIGNMENT = 2
    DRIVING_ARC = 3
    FINAL_ALIGNMENT = 4
    PAUSING_AFTER_FINAL_ALIGNMENT = 5
    REACHED_WAYPOINT_TARGET = 6
    ALL_WAYPOINTS_COMPLETE = 7
    STUCK_RECOVERY_LOCAL_MANEUVER = 8
    REQUESTING_GLOBAL_REPLAN = 9
    WAITING_FOR_NEW_PATH = 10

class TankWaypointNavigator(Node):
    def __init__(self):
        super().__init__('blimp_goto_node')

        self.declare_parameter('max_linear_speed', 0.015)
        self.declare_parameter('min_approach_linear_speed', 0.010)
        self.declare_parameter('max_angular_speed_align', 0.35)
        self.declare_parameter('max_angular_speed_drive', 0.03)
        self.declare_parameter('min_active_angular_speed', 0.06)
        self.declare_parameter('kp_linear', 0.2)
        self.declare_parameter('kp_angular_align', 1.0)
        self.declare_parameter('kp_angular_drive', 0.4)
        self.declare_parameter('waypoint_position_tolerance', 0.07)
        self.declare_parameter('initial_alignment_tolerance', 0.10)
        self.declare_parameter('driving_keep_alignment_tolerance', 0.35)
        self.declare_parameter('lookahead_distance_drive', 0.4)
        self.declare_parameter('final_orientation_tolerance', 0.08)
        self.declare_parameter('angular_command_deadzone', 0.03)
        self.declare_parameter('robot_base_frame', 'base_link')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('control_frequency', 10.0)
        self.declare_parameter('enable_final_orientation_at_last_waypoint', True)
        self.declare_parameter('pause_duration_after_turn', 0.5)
        self.declare_parameter('stuck_time_threshold', 4.0) 
        self.declare_parameter('local_costmap_topic', '/local_costmap/costmap')
        self.declare_parameter('enable_obstacle_avoidance', True)
        self.declare_parameter('robot_footprint_length', 0.35)
        self.declare_parameter('robot_footprint_width', 0.25)
        self.declare_parameter('collision_check_time_horizon', 0.75)
        self.declare_parameter('num_trajectory_check_points', 5)
        self.declare_parameter('replan_if_path_blocked', True)
        self.declare_parameter('path_check_lookahead_distance_m', 1.0)
        self.declare_parameter('path_check_resolution_m', 0.1)
        self.declare_parameter('path_is_blocked_threshold', 60)
        self.declare_parameter('max_local_stuck_attempts', 1)
        self.declare_parameter('replan_request_timeout_s', 10.0)
        self.declare_parameter('single_goal_topic', '/goal_pose')
        self.declare_parameter('path_topic', '/global_path')
        self.declare_parameter('temp_avoidance_topic', '/temp_avoidance_points')

        self.max_linear_speed=self.get_parameter('max_linear_speed').value
        self.min_approach_linear_speed=self.get_parameter('min_approach_linear_speed').value
        self.max_angular_speed_align=self.get_parameter('max_angular_speed_align').value
        self.max_angular_speed_drive=self.get_parameter('max_angular_speed_drive').value
        self.min_active_angular_speed=self.get_parameter('min_active_angular_speed').value
        self.kp_linear=self.get_parameter('kp_linear').value
        self.kp_angular_align=self.get_parameter('kp_angular_align').value
        self.kp_angular_drive=self.get_parameter('kp_angular_drive').value
        self.waypoint_pos_tol=self.get_parameter('waypoint_position_tolerance').value
        self.initial_align_tol=self.get_parameter('initial_alignment_tolerance').value
        self.driving_keep_align_tol=self.get_parameter('driving_keep_alignment_tolerance').value
        self.lookahead_dist_drive=self.get_parameter('lookahead_distance_drive').value
        self.final_orient_tol=self.get_parameter('final_orientation_tolerance').value
        self.angular_cmd_deadzone=self.get_parameter('angular_command_deadzone').value
        self.robot_base_frame=self.get_parameter('robot_base_frame').value
        self.map_frame=self.get_parameter('map_frame').value
        self.control_frequency=self.get_parameter('control_frequency').value
        self.control_period = 1.0 / self.control_frequency
        self.enable_final_orient_last_wp=self.get_parameter('enable_final_orientation_at_last_waypoint').value
        self.pause_duration=rclpyDuration(seconds=self.get_parameter('pause_duration_after_turn').value)
        self.stuck_time_threshold=self.get_parameter('stuck_time_threshold').value
        self.local_costmap_topic_name=self.get_parameter('local_costmap_topic').value
        self.enable_obs_avoid=self.get_parameter('enable_obstacle_avoidance').value
        self.robot_fp_len=self.get_parameter('robot_footprint_length').value
        self.robot_fp_wid=self.get_parameter('robot_footprint_width').value
        self.collision_check_time=self.get_parameter('collision_check_time_horizon').value
        self.num_traj_check_points=self.get_parameter('num_trajectory_check_points').value
        self.replan_if_path_blocked=self.get_parameter('replan_if_path_blocked').value
        self.path_check_lookahead_dist=self.get_parameter('path_check_lookahead_distance_m').value
        self.path_check_resolution=self.get_parameter('path_check_resolution_m').value
        self.path_blocked_threshold=self.get_parameter('path_is_blocked_threshold').value
        self.max_local_stuck_attempts=self.get_parameter('max_local_stuck_attempts').value
        self.replan_request_timeout_duration=rclpyDuration(seconds=self.get_parameter('replan_request_timeout_s').value)
        self.single_goal_topic_name=self.get_parameter('single_goal_topic').value
        self.path_topic_name=self.get_parameter('path_topic').value
        self.temp_avoidance_topic_name = self.get_parameter('temp_avoidance_topic').value

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.goal_republisher = self.create_publisher(PoseStamped, self.single_goal_topic_name, 5)
        self.temp_avoid_pub = self.create_publisher(Float32MultiArray, self.temp_avoidance_topic_name, 5)
        self.single_goal_sub = self.create_subscription(PoseStamped, self.single_goal_topic_name, self._on_single_goal_received, 10)
        self.path_sub = self.create_subscription(Path, self.path_topic_name, self._on_path_received, 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.current_target_pose: PoseStamped | None = None
        self.waypoint_list: list[PoseStamped] = []
        self.current_waypoint_index: int = -1
        self.robot_state = RobotState.IDLE
        self.action_start_time: rclpyTime | None = None # Can be None if not in timed state
        self.last_known_distance = float('inf')
        self.stuck_recovery_attempts_current_wp = 0
        self.next_state_after_pause = None
        self.ultimate_goal_pose: PoseStamped | None = None
        self.is_internal_replan_pending = False
        self.last_reported_blockage_point: Point | None = None
        self.local_costmap_data: np.ndarray | None = None
        self.local_costmap_grid: OccupancyGrid | None = None

        if self.enable_obs_avoid:
            costmap_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST, depth=1)
            self.costmap_sub = self.create_subscription(OccupancyGrid, self.local_costmap_topic_name, self._local_costmap_callback, costmap_qos)
            self.get_logger().info(f"Local obstacle avoidance features enabled, subscribing to {self.local_costmap_topic_name}")
        else:
            self.get_logger().info("Local obstacle avoidance features DISABLED.")
        self.control_timer = self.create_timer(self.control_period, self._control_loop)
        self.get_logger().info(f"{self.get_name()} started. Control Freq: {self.control_frequency} Hz.")

    def _on_single_goal_received(self, msg: PoseStamped):
        is_self_published_replan = False
        if self.is_internal_replan_pending and self.ultimate_goal_pose and \
           math.isclose(msg.pose.position.x, self.ultimate_goal_pose.pose.position.x, abs_tol=0.01) and \
           math.isclose(msg.pose.position.y, self.ultimate_goal_pose.pose.position.y, abs_tol=0.01):
            # It's highly likely our own replan goal if positions match closely
            is_self_published_replan = True
            self.get_logger().debug("Ignoring self-published goal for replan via _on_single_goal_received.")
            return # Do not process it as a new external goal

        if msg.header.frame_id != self.map_frame:
            self.get_logger().warn(f"Single goal in wrong frame '{msg.header.frame_id}', expected '{self.map_frame}'.")
            return
        
        self.get_logger().info("Received NEW EXTERNAL single goal. Resetting navigation state.")
        self.waypoint_list = [msg]
        self.current_waypoint_index = 0
        self.current_target_pose = msg
        self.ultimate_goal_pose = msg
        self.stuck_recovery_attempts_current_wp = 0
        self.is_internal_replan_pending = False 
        self.last_reported_blockage_point = None
        self._start_navigation_to_current_target("New external single goal received.")

    def _on_path_received(self, msg: Path):
        path_is_for_current_replan = False
        if self.robot_state == RobotState.WAITING_FOR_NEW_PATH:
            path_is_for_current_replan = True # Assume path received while waiting is for the replan
            self.is_internal_replan_pending = False # Clear the flag, path (or timeout) is resolving it

        if self.robot_state == RobotState.WAITING_FOR_NEW_PATH and self.action_start_time is not None:
            path_stamp_msg = msg.header.stamp
            request_time_msg = self.action_start_time.to_msg()
            path_is_older_or_same = False
            if path_stamp_msg.sec < request_time_msg.sec: path_is_older_or_same = True
            elif path_stamp_msg.sec == request_time_msg.sec and path_stamp_msg.nanosec <= request_time_msg.nanosec:
                path_is_older_or_same = True
            if path_is_older_or_same:
                self.get_logger().info(f"Received an old/same path (stamp: {path_stamp_msg.sec}.{path_stamp_msg.nanosec:09d}) " \
                                       f"while waiting for replan (request time: {request_time_msg.sec}.{request_time_msg.nanosec:09d}), ignoring.")
                # Do NOT clear is_internal_replan_pending here, still waiting for a NEWER path
                return

        if not msg.poses:
            self.get_logger().warn("Received empty path.")
            if path_is_for_current_replan:
                 self.get_logger().error("Global planner returned empty path after replan request. Mission failed for this goal.")
                 self._transition_state(RobotState.IDLE, "Empty path from replan.")
            return

        if msg.header.frame_id != self.map_frame:
            self.get_logger().warn(f"Path in wrong frame '{msg.header.frame_id}', expected '{self.map_frame}'.")
            return

        self.get_logger().info(f"Received new path with {len(msg.poses)} waypoints from {self.path_topic_name}.")
        self.waypoint_list = msg.poses
        self.current_waypoint_index = 0
        self.current_target_pose = self.waypoint_list[self.current_waypoint_index]

        if not path_is_for_current_replan: # If this is a new external path, update ultimate goal
            self.ultimate_goal_pose = self.waypoint_list[-1] if self.waypoint_list else None
            self.stuck_recovery_attempts_current_wp = 0 
            self.last_reported_blockage_point = None
        
        self._start_navigation_to_current_target(f"Starting/Resuming path: Waypoint {self.current_waypoint_index + 1}/{len(self.waypoint_list)}.")

    def _start_navigation_to_current_target(self, reason=""):
        if self.current_target_pose is None:
            self._transition_state(RobotState.IDLE, "No target pose.")
            return
        current_pose_map = self._get_current_pose_and_yaw()
        if current_pose_map:
            errors = self._calculate_errors(current_pose_map[0], current_pose_map[1], current_pose_map[2])
            if errors: self.last_known_distance = errors[0]
            else: self.last_known_distance = float('inf')
        else: self.last_known_distance = float('inf')
        self._transition_state(RobotState.ALIGNING, reason)
        self.stuck_recovery_attempts_current_wp = 0
        self.get_logger().info(f"Navigating to target: X={self.current_target_pose.pose.position.x:.2f}, Y={self.current_target_pose.pose.position.y:.2f}")

    def _get_current_pose_and_yaw(self) -> tuple[float, float, float] | None:
        try:
            transform = self.tf_buffer.lookup_transform(self.map_frame, self.robot_base_frame, rclpyTime(), rclpyDuration(seconds=0.2))
            trans, rot = transform.transform.translation, transform.transform.rotation
            _, _, yaw = euler_from_quaternion([rot.x, rot.y, rot.z, rot.w])
            return trans.x, trans.y, yaw
        except Exception as e:
            self.get_logger().warn(f"TF lookup ({self.map_frame} to {self.robot_base_frame}) failed: {e}", throttle_duration_sec=1.0)
            return None

    def _calculate_errors(self, current_x, current_y, current_yaw, target_pose: PoseStamped | None = None) -> tuple[float, float, float] | None:
        active_target_pose = target_pose if target_pose else self.current_target_pose
        if not active_target_pose: return None
        goal_x, goal_y = active_target_pose.pose.position.x, active_target_pose.pose.position.y
        goal_q = active_target_pose.pose.orientation
        _, _, goal_target_yaw = euler_from_quaternion([goal_q.x, goal_q.y, goal_q.z, goal_q.w])
        error_x, error_y = goal_x - current_x, goal_y - current_y
        distance_to_goal_position = math.hypot(error_x, error_y)
        angle_towards_goal_point = math.atan2(error_y, error_x)
        angle_error_to_reach_point = self._normalize_angle(angle_towards_goal_point - current_yaw)
        angle_error_to_final_orientation = self._normalize_angle(goal_target_yaw - current_yaw)
        return distance_to_goal_position, angle_error_to_reach_point, angle_error_to_final_orientation

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        while angle > math.pi: angle -= 2.0 * math.pi
        while angle < -math.pi: angle += 2.0 * math.pi
        return angle

    def _publish_cmd_vel(self, linear_x: float, angular_z: float):
        twist = Twist(); twist.linear.x = float(linear_x); twist.angular.z = float(angular_z)
        self.cmd_pub.publish(twist)

    def _transition_state(self, new_state, reason="", next_state_if_pausing=None):
        if self.robot_state != new_state or new_state in [RobotState.PAUSING_AFTER_ALIGNMENT, RobotState.PAUSING_AFTER_FINAL_ALIGNMENT, RobotState.WAITING_FOR_NEW_PATH]:
            state_names = {v: k for k, v in RobotState.__dict__.items() if not k.startswith('_') and isinstance(v, int)}
            old_state_name = state_names.get(self.robot_state, str(self.robot_state))
            new_state_name = state_names.get(new_state, str(new_state))
            log_msg = f"Transitioning: {old_state_name} -> {new_state_name}" + (f" (Reason: {reason})" if reason else "")
            self.get_logger().info(log_msg)
            self.robot_state = new_state
            self.action_start_time = self.get_clock().now()
            if new_state in [RobotState.PAUSING_AFTER_ALIGNMENT, RobotState.PAUSING_AFTER_FINAL_ALIGNMENT]:
                self.next_state_after_pause = next_state_if_pausing
            if new_state == RobotState.IDLE:
                self.current_target_pose = None; self.waypoint_list = []; self.current_waypoint_index = -1
                self.ultimate_goal_pose = None; self.stuck_recovery_attempts_current_wp = 0
                self.is_internal_replan_pending = False; self.last_reported_blockage_point = None

    def _local_costmap_callback(self, msg: OccupancyGrid):
        if not self.enable_obs_avoid: return
        self.local_costmap_grid = msg
        if self.local_costmap_grid and self.local_costmap_grid.info:
            self.local_costmap_data = np.array(msg.data, dtype=np.int8).reshape((self.local_costmap_grid.info.height, self.local_costmap_grid.info.width))
        else: self.local_costmap_data = None

    def _get_robot_pose_in_costmap_frame(self) -> tuple[float, float, float] | None:
        if self.local_costmap_grid is None or not self.local_costmap_grid.header.frame_id: return None
        try:
            transform = self.tf_buffer.lookup_transform(self.local_costmap_grid.header.frame_id, self.robot_base_frame, rclpyTime(), rclpyDuration(seconds=0.1))
            trans, rot = transform.transform.translation, transform.transform.rotation
            _, _, yaw = euler_from_quaternion([rot.x, rot.y, rot.z, rot.w])
            return trans.x, trans.y, yaw
        except Exception as e:
            self.get_logger().warn(f"TF lookup ({self.local_costmap_grid.header.frame_id} to {self.robot_base_frame}) failed: {e}", throttle_duration_sec=1.0)
            return None

    def _is_world_coord_in_local_obstacle(self, world_x: float, world_y: float, cost_threshold: int) -> tuple[bool, Point | None]:
        blocked_point_map_frame = Point(); blocked_point_map_frame.x = world_x; blocked_point_map_frame.y = world_y
        if self.local_costmap_data is None or self.local_costmap_grid is None or self.local_costmap_grid.info is None: return True, blocked_point_map_frame
        map_info = self.local_costmap_grid.info
        pt_map_frame_stamped = tf2_geometry_msgs.PointStamped()
        pt_map_frame_stamped.header.frame_id = self.map_frame
        pt_map_frame_stamped.point = blocked_point_map_frame
        try:
            if not self.local_costmap_grid.header.frame_id: return True, blocked_point_map_frame
            pt_lc_frame_stamped = self.tf_buffer.transform(pt_map_frame_stamped, self.local_costmap_grid.header.frame_id, timeout=rclpyDuration(seconds=0.1))
        except Exception: return True, blocked_point_map_frame
        lc_origin_x, lc_origin_y, lc_res = map_info.origin.position.x, map_info.origin.position.y, map_info.resolution
        lc_w, lc_h = map_info.width, map_info.height
        if lc_res == 0: return True, blocked_point_map_frame
        cell_x = int((pt_lc_frame_stamped.point.x - lc_origin_x) / lc_res)
        cell_y = int((pt_lc_frame_stamped.point.y - lc_origin_y) / lc_res)
        if 0 <= cell_x < lc_w and 0 <= cell_y < lc_h:
            if self.local_costmap_data[cell_y, cell_x] >= cost_threshold: return True, blocked_point_map_frame
            return False, None
        return True, blocked_point_map_frame

    def _check_footprint_collision_at_pose_lc(self, robot_cx_lc: float, robot_cy_lc: float, robot_yaw_lc: float) -> bool:
        if self.local_costmap_data is None or self.local_costmap_grid is None: return True
        map_info = self.local_costmap_grid.info
        lc_origin_x, lc_origin_y, lc_res = map_info.origin.position.x, map_info.origin.position.y, map_info.resolution
        lc_w, lc_h = map_info.width, map_info.height
        if lc_res == 0: return True
        half_len, half_wid = self.robot_fp_len / 2.0, self.robot_fp_wid / 2.0
        cos_y, sin_y = math.cos(robot_yaw_lc), math.sin(robot_yaw_lc)
        rel_pts_robot_frame = [(half_len, half_wid), (half_len, -half_wid), (-half_len, half_wid), (-half_len, -half_wid), (half_len, 0.0), (0.0,0.0)]
        for rx_robot, ry_robot in rel_pts_robot_frame:
            pt_x_lc = robot_cx_lc + (rx_robot * cos_y - ry_robot * sin_y)
            pt_y_lc = robot_cy_lc + (rx_robot * sin_y + ry_robot * cos_y)
            cell_x = int((pt_x_lc - lc_origin_x) / lc_res); cell_y = int((pt_y_lc - lc_origin_y) / lc_res)
            if 0 <= cell_x < lc_w and 0 <= cell_y < lc_h:
                if self.local_costmap_data[cell_y, cell_x] >= self.path_blocked_threshold: return True
            else: return True
        return False

    def _check_future_trajectory_collision_lc(self, current_cx_lc: float, current_cy_lc: float, current_cyaw_lc: float, linear_vel: float, angular_vel: float) -> bool:
        if self.local_costmap_data is None: return True
        if abs(linear_vel) < 0.005 and abs(angular_vel) < 0.01: return self._check_footprint_collision_at_pose_lc(current_cx_lc, current_cy_lc, current_cyaw_lc)
        dt = self.collision_check_time / self.num_traj_check_points
        tx_lc, ty_lc, tyaw_lc = current_cx_lc, current_cy_lc, current_cyaw_lc
        if self._check_footprint_collision_at_pose_lc(tx_lc, ty_lc, tyaw_lc): return True
        for _ in range(self.num_traj_check_points):
            if abs(angular_vel) < 1e-3:
                dist_increment = linear_vel * dt
                tx_lc += dist_increment * math.cos(tyaw_lc); ty_lc += dist_increment * math.sin(tyaw_lc)
            else:
                radius = linear_vel / angular_vel; delta_theta = angular_vel * dt
                tx_lc += radius * (math.sin(tyaw_lc + delta_theta) - math.sin(tyaw_lc))
                ty_lc += radius * (-math.cos(tyaw_lc + delta_theta) + math.cos(tyaw_lc))
                tyaw_lc = self._normalize_angle(tyaw_lc + delta_theta)
            if self._check_footprint_collision_at_pose_lc(tx_lc, ty_lc, tyaw_lc): return True
        return False

    def _is_global_path_segment_blocked(self) -> tuple[bool, Point | None]:
        if not self.replan_if_path_blocked or self.local_costmap_data is None or self.local_costmap_grid is None or not self.waypoint_list or self.current_waypoint_index < 0:
            return False, None
        current_robot_pose_map = self._get_current_pose_and_yaw()
        if not current_robot_pose_map: return False, None
        path_segment_map_frame: list[Point] = []; accumulated_dist = 0.0
        p_start_map = Point(); p_start_map.x, p_start_map.y = current_robot_pose_map[0], current_robot_pose_map[1]
        path_segment_map_frame.append(p_start_map)
        start_idx_on_global_path = self.current_waypoint_index
        if self.current_target_pose:
            dist_to_curr_target = math.hypot(p_start_map.x - self.current_target_pose.pose.position.x, p_start_map.y - self.current_target_pose.pose.position.y)
            if dist_to_curr_target < self.path_check_resolution and (self.current_waypoint_index + 1) < len(self.waypoint_list):
                start_idx_on_global_path = self.current_waypoint_index + 1
        for i in range(start_idx_on_global_path, len(self.waypoint_list)):
            wp_map = self.waypoint_list[i].pose.position
            dist_to_wp = math.hypot(wp_map.x - path_segment_map_frame[-1].x, wp_map.y - path_segment_map_frame[-1].y)
            if accumulated_dist + dist_to_wp > self.path_check_lookahead_dist:
                if dist_to_wp > 1e-3:
                    remaining_dist = self.path_check_lookahead_dist - accumulated_dist; ratio = remaining_dist / dist_to_wp
                    intermediate_pt_map = Point()
                    intermediate_pt_map.x = path_segment_map_frame[-1].x + ratio * (wp_map.x - path_segment_map_frame[-1].x)
                    intermediate_pt_map.y = path_segment_map_frame[-1].y + ratio * (wp_map.y - path_segment_map_frame[-1].y)
                    path_segment_map_frame.append(intermediate_pt_map)
                break
            path_segment_map_frame.append(Point(x=wp_map.x, y=wp_map.y, z=wp_map.z)) # Create new Point
            accumulated_dist += dist_to_wp
            if accumulated_dist >= self.path_check_lookahead_dist: break
        if len(path_segment_map_frame) < 2: return False, None
        interpolated_points_map_frame = []
        for i in range(len(path_segment_map_frame) - 1):
            p1, p2 = path_segment_map_frame[i], path_segment_map_frame[i+1]
            segment_len = math.hypot(p2.x - p1.x, p2.y - p1.y)
            if segment_len < 1e-3: continue
            num_steps = math.ceil(segment_len / self.path_check_resolution); num_steps = max(1, num_steps)
            for j in range(int(num_steps) + 1):
                ratio = j / num_steps; pt = Point(); pt.x = p1.x + ratio * (p2.x - p1.x); pt.y = p1.y + ratio * (p2.y - p1.y)
                interpolated_points_map_frame.append(pt)
        if not interpolated_points_map_frame: return False, None
        for map_pt in interpolated_points_map_frame:
            is_blocked, blocked_pt = self._is_world_coord_in_local_obstacle(map_pt.x, map_pt.y, self.path_blocked_threshold)
            if is_blocked:
                self.get_logger().warn(f"Global path segment point (map_frame {map_pt.x:.2f},{map_pt.y:.2f}) is blocked in local costmap.")
                return True, blocked_pt
        return False, None

    def _control_loop(self):
        current_time = self.get_clock().now()

        if self.robot_state == RobotState.REQUESTING_GLOBAL_REPLAN:
            self._publish_cmd_vel(0.0, 0.0)
            if self.ultimate_goal_pose:
                self.get_logger().info(f"Publishing ultimate goal to {self.single_goal_topic_name} for global replan.")
                if self.last_reported_blockage_point:
                    avoid_msg = Float32MultiArray(); avoid_msg.data = [self.last_reported_blockage_point.x, self.last_reported_blockage_point.y]
                    self.temp_avoid_pub.publish(avoid_msg)
                    self.get_logger().info(f"Published temp_avoidance_point: ({avoid_msg.data[0]:.2f}, {avoid_msg.data[1]:.2f})")
                replan_goal = PoseStamped(); replan_goal.header.stamp = current_time.to_msg(); replan_goal.header.frame_id = self.map_frame
                replan_goal.pose = self.ultimate_goal_pose.pose
                self.is_internal_replan_pending = True
                self.goal_republisher.publish(replan_goal)
                self._transition_state(RobotState.WAITING_FOR_NEW_PATH, "Waiting for global planner response.")
            else:
                self.get_logger().error("Cannot request replan, no ultimate_goal_pose stored.")
                self._transition_state(RobotState.IDLE, "Failed replan request (no ultimate goal).")
            return

        if self.robot_state == RobotState.WAITING_FOR_NEW_PATH:
            self._publish_cmd_vel(0.0, 0.0)
            if self.action_start_time and (current_time - self.action_start_time) > self.replan_request_timeout_duration:
                self.get_logger().warn("Timeout waiting for new path after replan request. Mission failed for this goal.")
                self.is_internal_replan_pending = False
                self._transition_state(RobotState.IDLE, "Replan timeout.")
            return
        
        if self.robot_state == RobotState.IDLE or self.current_target_pose is None :
             self._publish_cmd_vel(0.0, 0.0)
             if self.robot_state != RobotState.IDLE: self._transition_state(RobotState.IDLE, "No target pose.")
             return

        pose_data_map_frame = self._get_current_pose_and_yaw()
        if pose_data_map_frame is None:
            self.get_logger().warn("Cannot get robot pose in map frame.", throttle_duration_sec=2.0); self._publish_cmd_vel(0.0, 0.0); return
        current_x_map, current_y_map, current_yaw_map = pose_data_map_frame

        errors_to_actual_wp = self._calculate_errors(current_x_map, current_y_map, current_yaw_map)
        if errors_to_actual_wp is None:
            self._publish_cmd_vel(0.0,0.0); self._transition_state(RobotState.IDLE, "Error calculating errors to target."); return
        dist_err, angle_err_to_actual_point, angle_err_to_final_orient = errors_to_actual_wp

        if self.enable_obs_avoid and self.replan_if_path_blocked and self.robot_state in [RobotState.ALIGNING, RobotState.DRIVING_ARC, RobotState.PAUSING_AFTER_ALIGNMENT]:
            is_blocked, blocked_map_point = self._is_global_path_segment_blocked()
            if is_blocked:
                self._publish_cmd_vel(0.0, 0.0); self.last_reported_blockage_point = blocked_map_point
                self._transition_state(RobotState.REQUESTING_GLOBAL_REPLAN, "Global path segment blocked locally."); return

        if self.enable_obs_avoid and self.robot_state == RobotState.DRIVING_ARC:
            robot_pose_lc_frame = self._get_robot_pose_in_costmap_frame()
            if robot_pose_lc_frame:
                lc_x, lc_y, lc_yaw = robot_pose_lc_frame
                tentative_linear = CLIP(self.kp_linear * dist_err, 0, self.max_linear_speed)
                # Simplified steering for collision check projection
                steering_angle_for_cmd = angle_err_to_actual_point 
                # For more accuracy, could use lookahead point for steering_angle_for_cmd as in DRIVING_ARC main logic
                tentative_angular = 0.0
                if abs(steering_angle_for_cmd) > self.angular_cmd_deadzone: tentative_angular = CLIP(self.kp_angular_drive * steering_angle_for_cmd, -self.max_angular_speed_drive, self.max_angular_speed_drive)
                if tentative_linear > 0.005 and self._check_future_trajectory_collision_lc(lc_x, lc_y, lc_yaw, tentative_linear, tentative_angular):
                    self.get_logger().warn("Commanded trajectory leads to collision in local costmap.")
                    self._publish_cmd_vel(0.0, 0.0)
                    current_robot_map_pose_tuple = self._get_current_pose_and_yaw() # Re-fetch for accuracy
                    if current_robot_map_pose_tuple:
                        self.last_reported_blockage_point = Point(x=current_robot_map_pose_tuple[0], y=current_robot_map_pose_tuple[1], z=0.0)
                    else: self.last_reported_blockage_point = None
                    self._transition_state(RobotState.REQUESTING_GLOBAL_REPLAN, "Commanded trajectory collision."); return
        
        is_last_wp_in_current_path = self.waypoint_list and (self.current_waypoint_index == len(self.waypoint_list) - 1)
        is_ultimate_goal_target = False
        if self.current_target_pose and self.ultimate_goal_pose:
            dx = self.current_target_pose.pose.position.x - self.ultimate_goal_pose.pose.position.x
            dy = self.current_target_pose.pose.position.y - self.ultimate_goal_pose.pose.position.y
            if math.hypot(dx,dy) < 0.01 : is_ultimate_goal_target = True
        if dist_err < self.waypoint_pos_tol and self.robot_state not in [RobotState.REACHED_WAYPOINT_TARGET, RobotState.ALL_WAYPOINTS_COMPLETE, RobotState.FINAL_ALIGNMENT, RobotState.PAUSING_AFTER_FINAL_ALIGNMENT, RobotState.REQUESTING_GLOBAL_REPLAN, RobotState.WAITING_FOR_NEW_PATH, RobotState.IDLE]:
            self._publish_cmd_vel(0.0, 0.0)
            if (is_last_wp_in_current_path or is_ultimate_goal_target) and self.enable_final_orient_last_wp and abs(angle_err_to_final_orient) > self.final_orient_tol:
                self._transition_state(RobotState.FINAL_ALIGNMENT, f"Target pos reached (dist {dist_err:.3f}m), needs final orient.")
            else: self._transition_state(RobotState.REACHED_WAYPOINT_TARGET, f"Target pos reached (dist {dist_err:.3f}m).")
            return

        if self.robot_state == RobotState.ALIGNING:
            target_angle_error = angle_err_to_actual_point
            if abs(target_angle_error) > self.angular_cmd_deadzone:
                turn_speed = CLIP(self.kp_angular_align * target_angle_error, -self.max_angular_speed_align, self.max_angular_speed_align)
                if 0 < abs(turn_speed) < self.min_active_angular_speed: turn_speed = math.copysign(self.min_active_angular_speed, turn_speed)
                self._publish_cmd_vel(0.0, turn_speed)
            else: self._publish_cmd_vel(0.0, 0.0)
            if abs(target_angle_error) <= self.initial_align_tol:
                self._publish_cmd_vel(0.0, 0.0)
                pose_after_align = self._get_current_pose_and_yaw()
                if pose_after_align:
                    errors_for_drive = self._calculate_errors(pose_after_align[0], pose_after_align[1], pose_after_align[2])
                    if errors_for_drive: self.last_known_distance = errors_for_drive[0]
                self._transition_state(RobotState.PAUSING_AFTER_ALIGNMENT, f"Initial align complete (err {target_angle_error:.3f}rad).", RobotState.DRIVING_ARC)
        elif self.robot_state == RobotState.PAUSING_AFTER_ALIGNMENT or self.robot_state == RobotState.PAUSING_AFTER_FINAL_ALIGNMENT:
            self._publish_cmd_vel(0.0, 0.0)
            if self.action_start_time and (current_time - self.action_start_time) >= self.pause_duration:
                if self.next_state_after_pause is not None:
                    if self.next_state_after_pause == RobotState.DRIVING_ARC:
                        pose_after_pause = self._get_current_pose_and_yaw()
                        if pose_after_pause:
                            errors_for_drive = self._calculate_errors(pose_after_pause[0], pose_after_pause[1], pose_after_pause[2])
                            if errors_for_drive: self.last_known_distance = errors_for_drive[0]
                    self._transition_state(self.next_state_after_pause, "Pause complete.")
                else: self._transition_state(RobotState.IDLE, "Error in pause logic: no next state.")
        elif self.robot_state == RobotState.DRIVING_ARC:
            lookahead_target_pose = PoseStamped(); lookahead_target_pose.header.frame_id = self.map_frame 
            vec_to_wp_x = self.current_target_pose.pose.position.x - current_x_map
            vec_to_wp_y = self.current_target_pose.pose.position.y - current_y_map
            actual_lookahead_dist = min(self.lookahead_dist_drive, dist_err) 
            if dist_err > 1e-3: 
                lookahead_target_pose.pose.position.x = current_x_map + (vec_to_wp_x / dist_err) * actual_lookahead_dist
                lookahead_target_pose.pose.position.y = current_y_map + (vec_to_wp_y / dist_err) * actual_lookahead_dist
            else: 
                lookahead_target_pose.pose.position.x = self.current_target_pose.pose.position.x
                lookahead_target_pose.pose.position.y = self.current_target_pose.pose.position.y
            lookahead_target_pose.pose.orientation = self.current_target_pose.pose.orientation
            errors_to_lookahead = self._calculate_errors(current_x_map, current_y_map, current_yaw_map, target_pose=lookahead_target_pose)
            steering_angle_error = angle_err_to_actual_point 
            if errors_to_lookahead: _, steering_angle_error, _ = errors_to_lookahead
            else: self.get_logger().warn("Could not calculate errors to lookahead point.", throttle_duration_sec=1.0)
            if abs(steering_angle_error) > self.driving_keep_align_tol:
                self._publish_cmd_vel(0.0, 0.0)
                self._transition_state(RobotState.ALIGNING, f"Steering err to lookahead ({steering_angle_error:.3f}rad) > tol ({self.driving_keep_align_tol:.3f}rad). Re-aligning.")
                return
            linear_speed = CLIP(self.kp_linear * dist_err, 0.0, self.max_linear_speed)
            if dist_err < self.waypoint_pos_tol * 3: linear_speed = CLIP(linear_speed, 0, self.min_approach_linear_speed)
            elif 0 < linear_speed < self.min_approach_linear_speed * 0.8 and dist_err > self.waypoint_pos_tol : linear_speed = self.min_approach_linear_speed * 0.8
            angular_speed_cmd = 0.0
            if self.max_angular_speed_drive > 1e-3 and abs(steering_angle_error) > self.angular_cmd_deadzone :
                angular_speed_cmd = CLIP(self.kp_angular_drive * steering_angle_error, -self.max_angular_speed_drive, self.max_angular_speed_drive)
            self._publish_cmd_vel(linear_speed, angular_speed_cmd)
            if linear_speed >= self.min_approach_linear_speed * 0.5 and self.action_start_time:
                min_expected_progress_dist = linear_speed * self.control_period * 0.1 
                if dist_err >= (self.last_known_distance - min_expected_progress_dist):
                    if (current_time - self.action_start_time).nanoseconds / 1e9 > self.stuck_time_threshold:
                        self.get_logger().warn(f"Fallback: Stuck in DRIVING_ARC. dist_err={dist_err:.3f}, last_known_dist={self.last_known_distance:.3f}")
                        self._publish_cmd_vel(0.0, 0.0)
                        current_robot_map_pose_tuple = self._get_current_pose_and_yaw()
                        if current_robot_map_pose_tuple: self.last_reported_blockage_point = Point(x=current_robot_map_pose_tuple[0], y=current_robot_map_pose_tuple[1], z=0.0)
                        else: self.last_reported_blockage_point = None
                        self._transition_state(RobotState.STUCK_RECOVERY_LOCAL_MANEUVER, "Stuck in DRIVING_ARC (fallback)."); return
                else: self.last_known_distance = dist_err; self.action_start_time = current_time 
            elif self.action_start_time: self.action_start_time = current_time; self.last_known_distance = dist_err
        elif self.robot_state == RobotState.FINAL_ALIGNMENT:
            target_angle_error = angle_err_to_final_orient
            if abs(target_angle_error) > self.angular_cmd_deadzone:
                turn_speed = CLIP(self.kp_angular_align * target_angle_error, -self.max_angular_speed_align, self.max_angular_speed_align)
                if 0 < abs(turn_speed) < self.min_active_angular_speed: turn_speed = math.copysign(self.min_active_angular_speed, turn_speed)
                self._publish_cmd_vel(0.0, turn_speed)
            else: self._publish_cmd_vel(0.0, 0.0)
            if abs(target_angle_error) <= self.final_orient_tol:
                self._publish_cmd_vel(0.0, 0.0)
                self._transition_state(RobotState.PAUSING_AFTER_FINAL_ALIGNMENT, f"Final align complete (err {target_angle_error:.3f}rad).", RobotState.REACHED_WAYPOINT_TARGET)
        elif self.robot_state == RobotState.STUCK_RECOVERY_LOCAL_MANEUVER:
            self.get_logger().warn(f"Fallback: Local stuck recovery attempt {self.stuck_recovery_attempts_current_wp + 1}.")
            self.stuck_recovery_attempts_current_wp += 1
            if self.stuck_recovery_attempts_current_wp > self.max_local_stuck_attempts:
                self.get_logger().error("Fallback: Max local stuck attempts reached. Requesting global replan.")
                self._publish_cmd_vel(0.0,0.0)
                self._transition_state(RobotState.REQUESTING_GLOBAL_REPLAN, "Max local stuck attempts, fallback replan.")
            else:
                self.get_logger().info("Fallback: Attempting local stuck recovery via re-alignment.")
                self._transition_state(RobotState.ALIGNING, "Attempting local stuck recovery (re-align).")
            return 
        elif self.robot_state == RobotState.REACHED_WAYPOINT_TARGET:
            self.get_logger().info(f"Current target/waypoint {self.current_waypoint_index+1 if self.current_waypoint_index !=-1 else '(single goal)'} position processed.")
            self.last_reported_blockage_point = None
            if self.waypoint_list and self.current_waypoint_index < len(self.waypoint_list) - 1:
                self.current_waypoint_index += 1
                self.current_target_pose = self.waypoint_list[self.current_waypoint_index]
                self._start_navigation_to_current_target(f"Proceeding to waypoint {self.current_waypoint_index + 1}/{len(self.waypoint_list)}.")
            else: self._transition_state(RobotState.ALL_WAYPOINTS_COMPLETE, "All waypoints in path reached or single goal achieved.")
        elif self.robot_state == RobotState.ALL_WAYPOINTS_COMPLETE:
            self.get_logger().info("All waypoints navigated successfully! Mission complete.")
            self._publish_cmd_vel(0.0,0.0) 
            self._transition_state(RobotState.IDLE, "Path navigation finished.")

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = TankWaypointNavigator()
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node: node.get_logger().info(f"Ctrl-C pressed, shutting down {node.get_name()}.")
    # except Exception as e: # Comment out the generic Exception catcher during debugging type errors
    #     if node: node.get_logger().error(f"Unhandled exception in {node.get_name()}: {e}", exc_info=True)
    #     else: print(f"Unhandled exception before node init: {e}")
    finally:
        if node and rclpy.ok():
             if hasattr(node, '_publish_cmd_vel'): node._publish_cmd_vel(0.0, 0.0)
             if hasattr(node, 'destroy_node') and callable(node.destroy_node): node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()