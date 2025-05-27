#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration as rclpyDuration
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.time import Time as rclpyTime

from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path, OccupancyGrid
from tf2_ros import Buffer, TransformListener
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
    STUCK_RECOVERY = 8
    AVOIDING_OBSTACLE = 9

class TankWaypointNavigator(Node):
    def __init__(self):
        super().__init__('tank_waypoint_navigator')

        # --- Speed Parameters ---
        self.declare_parameter('max_linear_speed', 0.015)        # Increased from 0.012 to 0.015
        self.declare_parameter('min_approach_linear_speed', 0.010) # Increased from 0.008 to 0.010
        self.declare_parameter('max_angular_speed_align', 0.35)   # Increased from 0.30 to 0.35
        # For a tank that mostly turns in place, this should be low or 0.0
        # It controls how much it *tries* to turn while already moving forward in DRIVING_ARC.
        self.declare_parameter('max_angular_speed_drive', 0.03)  # Increased from 0.02 to 0.03
        self.declare_parameter('min_active_angular_speed', 0.06) # Increased from 0.05 to 0.06

        # --- Proportional Gains ---
        self.declare_parameter('kp_linear', 0.2)
        self.declare_parameter('kp_angular_align', 1.0)
        self.declare_parameter('kp_angular_drive', 0.4) # Only effective if max_angular_speed_drive > 0

        # --- Tolerances & Thresholds ---
        self.declare_parameter('waypoint_position_tolerance', 0.07) # How close to a WP to consider it "reached"
        self.declare_parameter('initial_alignment_tolerance', 0.10) # Radians for ALIGNING state
        # *** THIS IS KEY for looser following of a good path ***
        self.declare_parameter('driving_keep_alignment_tolerance', 0.35) # Radians (e.g., ~20 degrees). Tune this!
        self.declare_parameter('final_orientation_tolerance', 0.08)
        self.declare_parameter('angular_command_deadzone', 0.03)

        # --- Behavior Parameters ---
        self.declare_parameter('stuck_time_threshold', 3.0)
        self.declare_parameter('robot_base_frame', 'base_link')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('control_frequency', 10.0)
        self.declare_parameter('enable_final_orientation_at_last_waypoint', True)
        self.declare_parameter('pause_duration_after_turn', 0.5)
        # Lookahead for DRIVING_ARC. Helps smooth response to path curvature.
        self.declare_parameter('lookahead_distance_drive', 0.4) # Meters

        # --- Topic Names ---
        self.declare_parameter('single_goal_topic', '/goal_pose')
        self.declare_parameter('path_topic', '/global_path')

        # --- Obstacle Avoidance Parameters (Local Costmap - currently default to False) ---
        self.declare_parameter('local_costmap_topic', '/local_costmap/costmap')
        self.declare_parameter('robot_footprint_length', 0.35)
        self.declare_parameter('robot_footprint_width', 0.25)
        self.declare_parameter('collision_check_time_horizon', 0.75)
        self.declare_parameter('num_trajectory_check_points', 5)
        self.declare_parameter('enable_obstacle_avoidance', False) # Set to True if using local costmap
        self.declare_parameter('obstacle_avoidance_turn_speed_factor', 0.7)
        self.declare_parameter('obstacle_avoidance_duration_s', 2.5)

        # Get all parameters... (omitting for brevity, assume it's done as in your file)
        self.max_linear_speed = self.get_parameter('max_linear_speed').value
        self.min_approach_linear_speed = self.get_parameter('min_approach_linear_speed').value
        self.max_angular_speed_align = self.get_parameter('max_angular_speed_align').value
        self.max_angular_speed_drive = self.get_parameter('max_angular_speed_drive').value
        self.min_active_angular_speed = self.get_parameter('min_active_angular_speed').value
        self.kp_linear = self.get_parameter('kp_linear').value
        self.kp_angular_align = self.get_parameter('kp_angular_align').value
        self.kp_angular_drive = self.get_parameter('kp_angular_drive').value
        self.waypoint_pos_tol = self.get_parameter('waypoint_position_tolerance').value
        self.initial_align_tol = self.get_parameter('initial_alignment_tolerance').value
        self.driving_keep_align_tol = self.get_parameter('driving_keep_alignment_tolerance').value
        self.final_orient_tol = self.get_parameter('final_orientation_tolerance').value
        self.angular_cmd_deadzone = self.get_parameter('angular_command_deadzone').value
        self.stuck_time_threshold = self.get_parameter('stuck_time_threshold').value
        self.robot_base_frame = self.get_parameter('robot_base_frame').value
        self.map_frame = self.get_parameter('map_frame').value
        self.enable_final_orient_last_wp = self.get_parameter('enable_final_orientation_at_last_waypoint').value
        self.pause_duration = rclpyDuration(seconds=self.get_parameter('pause_duration_after_turn').value)
        self.lookahead_dist_drive = self.get_parameter('lookahead_distance_drive').value
        self.single_goal_topic_name = self.get_parameter('single_goal_topic').value
        self.path_topic_name = self.get_parameter('path_topic').value
        self.control_frequency = self.get_parameter('control_frequency').value
        self.control_period = 1.0 / self.control_frequency
        self.local_costmap_topic_name = self.get_parameter('local_costmap_topic').value
        self.robot_fp_len = self.get_parameter('robot_footprint_length').value
        self.robot_fp_wid = self.get_parameter('robot_footprint_width').value
        self.collision_check_time = self.get_parameter('collision_check_time_horizon').value
        self.num_traj_check_points = self.get_parameter('num_trajectory_check_points').value
        self.enable_obs_avoid = self.get_parameter('enable_obstacle_avoidance').value
        self.avoid_turn_factor = self.get_parameter('obstacle_avoidance_turn_speed_factor').value
        self.avoid_duration = rclpyDuration(seconds=self.get_parameter('obstacle_avoidance_duration_s').value)

        # ... (rest of __init__ is the same as your last version) ...
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.single_goal_sub = self.create_subscription(
            PoseStamped, self.single_goal_topic_name, self._on_single_goal_received, 10)
        self.path_sub = self.create_subscription(
            Path, self.path_topic_name, self._on_path_received, 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.current_target_pose: PoseStamped | None = None
        self.waypoint_list: list[PoseStamped] = []
        self.current_waypoint_index: int = -1
        self.robot_state = RobotState.IDLE
        self.action_start_time = self.get_clock().now()
        self.last_known_distance = float('inf')
        self.stuck_recovery_attempts = 0
        self.next_state_after_pause = None

        self.local_costmap_data: np.ndarray | None = None
        self.local_costmap_grid: OccupancyGrid | None = None
        self.avoidance_active_until = self.get_clock().now()

        if self.enable_obs_avoid:
            costmap_qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
                history=HistoryPolicy.KEEP_LAST, depth=1 )
            self.costmap_sub = self.create_subscription(
                OccupancyGrid, self.local_costmap_topic_name, self._local_costmap_callback, costmap_qos)
            self.get_logger().info(f"Obstacle avoidance enabled, subscribing to {self.local_costmap_topic_name}")
        else:
            self.get_logger().info("Obstacle avoidance DISABLED.")

        self.control_timer = self.create_timer(self.control_period, self._control_loop)
        self.get_logger().info(f"{self.get_name()} started. Ctrl Freq: {self.control_frequency} Hz.")
        self.get_logger().info(f"**Critical Tolerances: Driving Align Tol: {self.driving_keep_align_tol:.2f} rad ({math.degrees(self.driving_keep_align_tol):.1f} deg), "
                               f"Lookahead: {self.lookahead_dist_drive:.2f}m, Max Angular Drive: {self.max_angular_speed_drive:.2f} rad/s**")


    # _on_single_goal_received, _on_path_received, _start_navigation_to_current_target
    # _get_current_pose_and_yaw, _calculate_errors, _normalize_angle, _publish_cmd_vel,
    # _transition_state, _local_costmap_callback, _get_robot_pose_in_costmap_frame,
    # _is_world_coord_occupied, _check_footprint_collision_at_pose, _check_future_trajectory_collision
    # ARE THE SAME AS YOUR PREVIOUSLY WORKING VERSION (the one I sent with lookahead integrated)

    # --- THE _control_loop REMAINS THE SAME AS THE PREVIOUS VERSION I SENT, which includes: ---
    # 1. Calculation of errors_to_actual_wp
    # 2. Obstacle avoidance check (if enabled)
    # 3. Universal target position reached check
    # 4. State machine logic:
    #    - IDLE
    #    - ALIGNING (uses angle_err_to_actual_point)
    #    - PAUSING_AFTER_ALIGNMENT / PAUSING_AFTER_FINAL_ALIGNMENT
    #    - DRIVING_ARC:
    #        - Calculates lookahead_target_pose
    #        - Calculates errors_to_lookahead to get steering_angle_error
    #        - Checks if abs(steering_angle_error) > self.driving_keep_align_tol to trigger re-ALIGNING
    #        - Calculates linear_speed based on dist_err (to actual waypoint)
    #        - Calculates angular_speed_cmd based on steering_angle_error and self.max_angular_speed_drive
    #        - Stuck detection based on dist_err
    #    - AVOIDING_OBSTACLE
    #    - FINAL_ALIGNMENT
    #    - STUCK_RECOVERY
    #    - REACHED_WAYPOINT_TARGET
    #    - ALL_WAYPOINTS_COMPLETE
    # --- NO CHANGES NEEDED TO THE _control_loop LOGIC ITSELF FROM THE LAST VERSION I PROVIDED ---
    # The key is that the *parameters* it uses (driving_keep_align_tol, lookahead_dist_drive, max_angular_speed_drive)
    # and the *path it receives* (now from the non-simplifying global planner) will change its behavior.

    # Paste the full _control_loop and other helper methods from the previous version
    # (the one that had the lookahead logic correctly implemented in DRIVING_ARC) here.
    # For brevity, I'm not repeating the entire _control_loop block if it's identical
    # to the one I sent that introduced the lookahead in DRIVING_ARC.
    # The crucial part is that the *parameter values* are now different.

    # ... [Paste the _control_loop and other helper methods from the previous 'tank_waypoint_navigator.py' I sent] ...
    # Make sure to include:
    # _on_single_goal_received, _on_path_received, _start_navigation_to_current_target
    # _get_current_pose_and_yaw, _calculate_errors (the one that takes optional target_pose)
    # _normalize_angle, _publish_cmd_vel, _transition_state
    # _local_costmap_callback, _get_robot_pose_in_costmap_frame, _is_world_coord_occupied,
    # _check_footprint_collision_at_pose, _check_future_trajectory_collision
    # And the entire _control_loop with its state machine.
    # The version I sent previously for TankWaypointNavigator with the lookahead logic in DRIVING_ARC is correct.
    # The change is now primarily in the default parameter values tuned in __init__ and using the
    # non-simplifying global planner.

    # --- Placeholder for the rest of the methods from previous correct version ---
    def _on_single_goal_received(self, msg: PoseStamped): # Copied from previous good version
        if msg.header.frame_id != self.map_frame:
            self.get_logger().warn(f"Single goal in wrong frame '{msg.header.frame_id}', expected '{self.map_frame}'.")
            return
        self.get_logger().info("Received single goal. Clearing path, navigating directly.")
        self.waypoint_list = [msg]
        self.current_waypoint_index = 0
        self.current_target_pose = msg
        self._start_navigation_to_current_target("Single goal received as path.")

    def _on_path_received(self, msg: Path): # Copied from previous good version
        if not msg.poses:
            self.get_logger().warn("Received empty path.")
            return
        if msg.header.frame_id != self.map_frame:
            self.get_logger().warn(f"Path in wrong frame '{msg.header.frame_id}', expected '{self.map_frame}'.")
            return
        self.get_logger().info(f"Received new path with {len(msg.poses)} waypoints from {self.path_topic_name}.")
        self.waypoint_list = msg.poses
        self.current_waypoint_index = 0
        self.current_target_pose = self.waypoint_list[self.current_waypoint_index]
        self._start_navigation_to_current_target(f"Starting path: Waypoint {self.current_waypoint_index + 1}/{len(self.waypoint_list)}.")

    def _start_navigation_to_current_target(self, reason=""): # Copied from previous good version
        if self.current_target_pose is None:
            self._transition_state(RobotState.IDLE, "No target pose.")
            return
        current_pose_map = self._get_current_pose_and_yaw()
        if current_pose_map and self.current_target_pose:
            errors = self._calculate_errors(current_pose_map[0], current_pose_map[1], current_pose_map[2])
            if errors: self.last_known_distance = errors[0]
            else: self.last_known_distance = float('inf')
        else: self.last_known_distance = float('inf')
            
        self._transition_state(RobotState.ALIGNING, reason)
        self.stuck_recovery_attempts = 0
        self.get_logger().info(
             f"Navigating to target: X={self.current_target_pose.pose.position.x:.2f}, Y={self.current_target_pose.pose.position.y:.2f}")

    def _get_current_pose_and_yaw(self) -> tuple[float, float, float] | None: # Copied
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame, self.robot_base_frame, rclpyTime(), rclpyDuration(seconds=0.2))
            trans = transform.transform.translation
            rot = transform.transform.rotation
            _, _, yaw = euler_from_quaternion([rot.x, rot.y, rot.z, rot.w])
            return trans.x, trans.y, yaw
        except Exception as e:
            self.get_logger().warn(f"TF lookup ({self.map_frame} to {self.robot_base_frame}) failed: {e}", throttle_duration_sec=1.0)
            return None

    def _calculate_errors(self, current_x, current_y, current_yaw, target_pose: PoseStamped | None = None) -> tuple[float, float, float] | None: # Copied
        active_target_pose = target_pose if target_pose else self.current_target_pose
        if not active_target_pose: return None
        goal_x = active_target_pose.pose.position.x
        goal_y = active_target_pose.pose.position.y
        goal_q = active_target_pose.pose.orientation
        _, _, goal_target_yaw = euler_from_quaternion([goal_q.x, goal_q.y, goal_q.z, goal_q.w])
        error_x = goal_x - current_x
        error_y = goal_y - current_y
        distance_to_goal_position = math.hypot(error_x, error_y)
        angle_towards_goal_point = math.atan2(error_y, error_x)
        angle_error_to_reach_point = self._normalize_angle(angle_towards_goal_point - current_yaw)
        angle_error_to_final_orientation = self._normalize_angle(goal_target_yaw - current_yaw)
        return distance_to_goal_position, angle_error_to_reach_point, angle_error_to_final_orientation

    @staticmethod
    def _normalize_angle(angle: float) -> float: # Copied
        while angle > math.pi: angle -= 2.0 * math.pi
        while angle < -math.pi: angle += 2.0 * math.pi
        return angle

    def _publish_cmd_vel(self, linear_x: float, angular_z: float): # Copied
        twist = Twist()
        twist.linear.x = float(linear_x)
        twist.angular.z = float(angular_z)
        self.cmd_pub.publish(twist)

    def _transition_state(self, new_state, reason="", next_state_if_pausing=None): # Copied
        if self.robot_state != new_state or \
           (new_state in [RobotState.PAUSING_AFTER_ALIGNMENT, RobotState.PAUSING_AFTER_FINAL_ALIGNMENT, RobotState.AVOIDING_OBSTACLE]):
            state_names = {v: k for k, v in RobotState.__dict__.items() if not k.startswith('_') and isinstance(v, int)}
            old_state_name = state_names.get(self.robot_state, str(self.robot_state))
            new_state_name = state_names.get(new_state, str(new_state))
            log_msg = f"Transitioning: {old_state_name} -> {new_state_name}"
            if reason: log_msg += f" (Reason: {reason})"
            self.get_logger().info(log_msg)
            self.robot_state = new_state
            self.action_start_time = self.get_clock().now()
            if new_state in [RobotState.PAUSING_AFTER_ALIGNMENT, RobotState.PAUSING_AFTER_FINAL_ALIGNMENT]:
                self.next_state_after_pause = next_state_if_pausing

    def _local_costmap_callback(self, msg: OccupancyGrid): # Copied
        if not self.enable_obs_avoid: return
        self.local_costmap_grid = msg
        if self.local_costmap_grid and self.local_costmap_grid.info:
            self.local_costmap_data = np.array(msg.data, dtype=np.int8).reshape(
                (self.local_costmap_grid.info.height, self.local_costmap_grid.info.width)
            )
        else: self.local_costmap_data = None

    def _get_robot_pose_in_costmap_frame(self) -> tuple[float, float, float] | None: # Copied
        if self.local_costmap_grid is None or not self.local_costmap_grid.header.frame_id: return None
        try:
            transform = self.tf_buffer.lookup_transform(
                self.local_costmap_grid.header.frame_id, self.robot_base_frame,
                rclpyTime(), rclpyDuration(seconds=0.1))
            trans, rot = transform.transform.translation, transform.transform.rotation
            _, _, yaw = euler_from_quaternion([rot.x, rot.y, rot.z, rot.w])
            return trans.x, trans.y, yaw
        except Exception as e:
            self.get_logger().warn(f"TF lookup ({self.local_costmap_grid.header.frame_id} to {self.robot_base_frame}) failed: {e}", throttle_duration_sec=1.0)
            return None

    def _is_world_coord_occupied(self, world_x: float, world_y: float) -> bool: # Copied
        if self.local_costmap_data is None or self.local_costmap_grid is None or self.local_costmap_grid.info is None: return True
        map_info = self.local_costmap_grid.info
        origin_x, origin_y, res, w, h = map_info.origin.position.x, map_info.origin.position.y, map_info.resolution, map_info.width, map_info.height
        if res == 0: return True
        cell_x, cell_y = int((world_x - origin_x) / res), int((world_y - origin_y) / res)
        if 0 <= cell_x < w and 0 <= cell_y < h: return self.local_costmap_data[cell_y, cell_x] >= 50
        return True

    def _check_footprint_collision_at_pose(self, robot_cx: float, robot_cy: float, robot_yaw: float) -> bool: # Copied
        if self.local_costmap_data is None: return True
        half_len, half_wid = self.robot_fp_len / 2.0, self.robot_fp_wid / 2.0
        cos_y, sin_y = math.cos(robot_yaw), math.sin(robot_yaw)
        rel_pts = [(half_len, half_wid), (half_len, -half_wid), (-half_len, half_wid), (-half_len, -half_wid), (half_len, 0.0), (0.0, 0.0)]
        for rx, ry in rel_pts:
            wx, wy = robot_cx + (rx * cos_y - ry * sin_y), robot_cy + (rx * sin_y + ry * cos_y)
            if self._is_world_coord_occupied(wx, wy): return True
        return False

    def _check_future_trajectory_collision(self, current_cx: float, current_cy: float, current_cyaw: float,
                                         linear_vel: float, angular_vel: float) -> bool: # Copied
        if self.local_costmap_data is None: return True
        if abs(linear_vel) < 0.005 and abs(angular_vel) < 0.01: return self._check_footprint_collision_at_pose(current_cx, current_cy, current_cyaw)
        projection_time, dt = self.collision_check_time, self.collision_check_time / self.num_traj_check_points
        tx, ty, tyaw = current_cx, current_cy, current_cyaw
        if self._check_footprint_collision_at_pose(tx, ty, tyaw): return True
        for _ in range(self.num_traj_check_points):
            if abs(angular_vel) < 1e-3:
                dist_inc = linear_vel * dt
                tx += dist_inc * math.cos(tyaw); ty += dist_inc * math.sin(tyaw)
            else:
                radius = linear_vel / angular_vel; d_theta = angular_vel * dt
                tx += radius * (math.sin(tyaw + d_theta) - math.sin(tyaw))
                ty += radius * (-math.cos(tyaw + d_theta) + math.cos(tyaw))
                tyaw = self._normalize_angle(tyaw + d_theta)
            if self._check_footprint_collision_at_pose(tx, ty, tyaw): return True
        return False

    def _control_loop(self): # Copied from the previous good version with lookahead
        if self.current_target_pose is None or self.robot_state == RobotState.IDLE:
            self._publish_cmd_vel(0.0, 0.0)
            return

        pose_data_map_frame = self._get_current_pose_and_yaw()
        if pose_data_map_frame is None:
            self._publish_cmd_vel(0.0, 0.0)
            return
        current_x_map, current_y_map, current_yaw_map = pose_data_map_frame

        errors_to_actual_wp = self._calculate_errors(current_x_map, current_y_map, current_yaw_map)
        if errors_to_actual_wp is None:
            self._publish_cmd_vel(0.0,0.0)
            self._transition_state(RobotState.IDLE, "Error: Target pose missing or error calculation failed for actual WP.")
            return
        dist_err, angle_err_to_actual_point, angle_err_to_final_orient = errors_to_actual_wp
        current_time = self.get_clock().now()

        if self.enable_obs_avoid and self.robot_state == RobotState.DRIVING_ARC and \
           self.local_costmap_data is not None and self.local_costmap_grid is not None and self.local_costmap_grid.info is not None:
            pose_in_costmap_frame = self._get_robot_pose_in_costmap_frame()
            if pose_in_costmap_frame:
                cm_x, cm_y, cm_yaw = pose_in_costmap_frame
                tentative_linear = CLIP(self.kp_linear * dist_err, 0.0, self.max_linear_speed)
                if dist_err < self.waypoint_pos_tol * 3 and tentative_linear > self.min_approach_linear_speed:
                    tentative_linear = self.min_approach_linear_speed
                tentative_angular = 0.0
                if abs(angle_err_to_actual_point) > self.angular_cmd_deadzone:
                    tentative_angular = CLIP(self.kp_angular_drive * angle_err_to_actual_point, -self.max_angular_speed_drive, self.max_angular_speed_drive)
                if tentative_linear > 0.005 and self._check_future_trajectory_collision(cm_x, cm_y, cm_yaw, tentative_linear, tentative_angular):
                    self.get_logger().warn("Predicted collision. Transitioning to AVOIDING_OBSTACLE.")
                    self._publish_cmd_vel(0.0, 0.0)
                    self.avoidance_active_until = current_time + self.avoid_duration
                    self._transition_state(RobotState.AVOIDING_OBSTACLE, "Predicted collision.")
                    return
        
        is_last_wp = self.waypoint_list and (self.current_waypoint_index == len(self.waypoint_list) - 1)
        if dist_err < self.waypoint_pos_tol and \
           self.robot_state not in [RobotState.REACHED_WAYPOINT_TARGET, RobotState.ALL_WAYPOINTS_COMPLETE,
                                    RobotState.FINAL_ALIGNMENT, RobotState.PAUSING_AFTER_FINAL_ALIGNMENT,
                                    RobotState.IDLE, RobotState.AVOIDING_OBSTACLE]:
            self._publish_cmd_vel(0.0, 0.0)
            if is_last_wp and self.enable_final_orient_last_wp and abs(angle_err_to_final_orient) > self.final_orient_tol:
                self._transition_state(RobotState.FINAL_ALIGNMENT, f"Last WP pos reached (dist {dist_err:.3f}m), needs final orient.")
            else:
                self._transition_state(RobotState.REACHED_WAYPOINT_TARGET, f"Target pos reached (dist {dist_err:.3f}m).")
            return

        if self.robot_state == RobotState.IDLE:
            pass 
        elif self.robot_state == RobotState.ALIGNING:
            target_angle_error = angle_err_to_actual_point
            if abs(target_angle_error) > self.angular_cmd_deadzone:
                turn_speed = CLIP(self.kp_angular_align * target_angle_error, -self.max_angular_speed_align, self.max_angular_speed_align)
                if 0 < abs(turn_speed) < self.min_active_angular_speed: turn_speed = math.copysign(self.min_active_angular_speed, turn_speed)
                self._publish_cmd_vel(0.0, turn_speed)
            else: self._publish_cmd_vel(0.0, 0.0)
            if abs(target_angle_error) <= self.initial_align_tol:
                self._publish_cmd_vel(0.0, 0.0)
                current_pose_map_after_align = self._get_current_pose_and_yaw()
                if current_pose_map_after_align and self.current_target_pose:
                    errors_for_drive = self._calculate_errors(current_pose_map_after_align[0], current_pose_map_after_align[1], current_pose_map_after_align[2])
                    if errors_for_drive: self.last_known_distance = errors_for_drive[0]
                self._transition_state(RobotState.PAUSING_AFTER_ALIGNMENT, f"Initial align complete (err {target_angle_error:.3f}rad).", RobotState.DRIVING_ARC)

        elif self.robot_state == RobotState.PAUSING_AFTER_ALIGNMENT or self.robot_state == RobotState.PAUSING_AFTER_FINAL_ALIGNMENT:
            self._publish_cmd_vel(0.0, 0.0)
            if (current_time - self.action_start_time) >= self.pause_duration:
                if self.next_state_after_pause is not None:
                    if self.next_state_after_pause == RobotState.DRIVING_ARC:
                        current_pose_map_after_pause = self._get_current_pose_and_yaw()
                        if current_pose_map_after_pause and self.current_target_pose:
                            errors_for_drive = self._calculate_errors(current_pose_map_after_pause[0], current_pose_map_after_pause[1], current_pose_map_after_pause[2])
                            if errors_for_drive: self.last_known_distance = errors_for_drive[0]
                    self._transition_state(self.next_state_after_pause, "Pause complete.")
                else: self._transition_state(RobotState.IDLE, "Error in pause logic.")

        elif self.robot_state == RobotState.DRIVING_ARC:
            lookahead_target_pose = PoseStamped()
            lookahead_target_pose.header = self.current_target_pose.header
            vec_to_wp_x = self.current_target_pose.pose.position.x - current_x_map
            vec_to_wp_y = self.current_target_pose.pose.position.y - current_y_map
            actual_lookahead_dist = min(self.lookahead_dist_drive, dist_err) 
            if dist_err > 0.01:
                lookahead_target_pose.pose.position.x = current_x_map + (vec_to_wp_x / dist_err) * actual_lookahead_dist
                lookahead_target_pose.pose.position.y = current_y_map + (vec_to_wp_y / dist_err) * actual_lookahead_dist
            else:
                lookahead_target_pose.pose.position.x = self.current_target_pose.pose.position.x
                lookahead_target_pose.pose.position.y = self.current_target_pose.pose.position.y
            lookahead_target_pose.pose.orientation = self.current_target_pose.pose.orientation

            errors_to_lookahead = self._calculate_errors(current_x_map, current_y_map, current_yaw_map, target_pose=lookahead_target_pose)
            steering_angle_error = angle_err_to_actual_point # Fallback
            if errors_to_lookahead:
                _, steering_angle_error, _ = errors_to_lookahead
            else:
                self.get_logger().warn("Could not calculate errors to lookahead point. Using actual waypoint for steering check.", throttle_duration_sec=1.0)

            if abs(steering_angle_error) > self.driving_keep_align_tol:
                self._publish_cmd_vel(0.0, 0.0)
                self._transition_state(RobotState.ALIGNING, 
                                       f"Steering err to lookahead ({steering_angle_error:.3f}rad) > tol ({self.driving_keep_align_tol:.3f}rad). Re-aligning to actual WP.")
                return

            linear_speed = CLIP(self.kp_linear * dist_err, 0.0, self.max_linear_speed)
            if dist_err < self.waypoint_pos_tol * 3 and linear_speed > self.min_approach_linear_speed:
                linear_speed = self.min_approach_linear_speed
            elif 0 < linear_speed < self.min_approach_linear_speed * 0.8 and dist_err > self.waypoint_pos_tol:
                 linear_speed = self.min_approach_linear_speed * 0.8
            
            angular_speed_cmd = 0.0
            if self.max_angular_speed_drive > 0.001 and abs(steering_angle_error) > self.angular_cmd_deadzone : # Only apply if allowed and outside deadzone
                angular_speed_cmd = CLIP(self.kp_angular_drive * steering_angle_error, 
                                         -self.max_angular_speed_drive, self.max_angular_speed_drive)
            self._publish_cmd_vel(linear_speed, angular_speed_cmd)

            if linear_speed >= self.min_approach_linear_speed * 0.5:
                min_expected_progress_dist = max(linear_speed, self.min_approach_linear_speed * 0.5) * self.control_period * 0.1
                if dist_err >= (self.last_known_distance - min_expected_progress_dist):
                    if (current_time - self.action_start_time).nanoseconds / 1e9 > self.stuck_time_threshold:
                        self.get_logger().warn(f"Stuck in DRIVING_ARC. dist_err={dist_err:.3f}, last_known_dist={self.last_known_distance:.3f}")
                        self._publish_cmd_vel(0.0, 0.0)
                        self._transition_state(RobotState.STUCK_RECOVERY, "Stuck in DRIVING_ARC.")
                        return
                else: 
                    self.last_known_distance = dist_err; self.action_start_time = current_time 
            else: 
                self.action_start_time = current_time; self.last_known_distance = dist_err

        elif self.robot_state == RobotState.AVOIDING_OBSTACLE:
            avoid_turn_dir = -1.0 if angle_err_to_actual_point > 0 else 1.0 
            if abs(angle_err_to_actual_point) < 0.1: avoid_turn_dir = 1.0 
            angular_vel_avoid = avoid_turn_dir * self.max_angular_speed_align * self.avoid_turn_factor
            self._publish_cmd_vel(0.0, angular_vel_avoid)
            if current_time >= self.avoidance_active_until:
                self._publish_cmd_vel(0.0, 0.0)
                self._transition_state(RobotState.ALIGNING, "Avoidance time up, re-align.")
            else: 
                pose_in_cm = self._get_robot_pose_in_costmap_frame()
                if pose_in_cm and not self._check_future_trajectory_collision(pose_in_cm[0], pose_in_cm[1], pose_in_cm[2], self.min_approach_linear_speed * 0.5, 0.0):
                    self._publish_cmd_vel(0.0, 0.0)
                    self._transition_state(RobotState.ALIGNING, "Avoidance turn may have cleared path.")

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

        elif self.robot_state == RobotState.STUCK_RECOVERY:
            self.get_logger().warn(f"Stuck recovery attempt {self.stuck_recovery_attempts + 1}.")
            self.stuck_recovery_attempts += 1
            if self.stuck_recovery_attempts > 2:
                self.get_logger().error("Multiple stuck recovery attempts failed. Giving up on current target.")
                self._publish_cmd_vel(0.0,0.0)
                if self.waypoint_list and self.current_waypoint_index < len(self.waypoint_list) - 1:
                    self.get_logger().info("Skipping current waypoint due to failed stuck recovery.")
                    self.current_waypoint_index += 1
                    self.current_target_pose = self.waypoint_list[self.current_waypoint_index]
                    self._start_navigation_to_current_target(f"Stuck recovery failed, proceeding to waypoint {self.current_waypoint_index + 1}/{len(self.waypoint_list)}.")
                else:
                    self.current_target_pose = None; self.waypoint_list = []; self.current_waypoint_index = -1
                    self._transition_state(RobotState.IDLE, "Stuck recovery failed. No more waypoints.")
            else: 
                self._transition_state(RobotState.ALIGNING, "Attempting stuck recovery via re-alignment.")

        elif self.robot_state == RobotState.REACHED_WAYPOINT_TARGET:
            self.get_logger().info(f"Current target/waypoint {self.current_waypoint_index+1 if self.current_waypoint_index !=-1 else '(single goal)'} position processed.")
            if self.waypoint_list and self.current_waypoint_index < len(self.waypoint_list) - 1:
                self.current_waypoint_index += 1
                self.current_target_pose = self.waypoint_list[self.current_waypoint_index]
                self._start_navigation_to_current_target(f"Proceeding to waypoint {self.current_waypoint_index + 1}/{len(self.waypoint_list)}.")
            else:
                self._transition_state(RobotState.ALL_WAYPOINTS_COMPLETE, "All waypoints in path reached or single goal achieved.")
        
        elif self.robot_state == RobotState.ALL_WAYPOINTS_COMPLETE:
            self.get_logger().info("All waypoints navigated successfully! Mission complete.")
            self._publish_cmd_vel(0.0,0.0) 
            self._transition_state(RobotState.IDLE, "Path navigation finished.")
            self.current_target_pose = None; self.waypoint_list = []; self.current_waypoint_index = -1

def main(args=None): # Copied
    rclpy.init(args=args)
    node = None
    try:
        node = TankWaypointNavigator()
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node: node.get_logger().info(f"Ctrl-C pressed, shutting down {node.get_name()}.")
    except Exception as e:
        if node: node.get_logger().error(f"Unhandled exception in {node.get_name()}: {e}", exc_info=True)
        else: print(f"Unhandled exception before node init: {e}")
    finally:
        if node and rclpy.ok():
             node.get_logger().info(f"{node.get_name()} ensuring robot is stopped before exiting.")
             node._publish_cmd_vel(0.0, 0.0)
             node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()