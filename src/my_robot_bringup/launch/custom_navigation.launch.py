import os
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
    TimerAction,
)
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.conditions import IfCondition


def generate_launch_description():
    # --- Define Package Shares ---
    pkg_my_robot_bringup = get_package_share_directory("my_robot_bringup")
    pkg_custom_nav = get_package_share_directory(
        "my_robot_driver"
    )  # Ensure this is your navigation scripts package

    # --- Launch Arguments ---
    use_sim_time = LaunchConfiguration("use_sim_time", default="false")
    log_level = LaunchConfiguration("log_level", default="info")
    use_rviz = LaunchConfiguration("use_rviz", default="true")

    rtabmap_slam_launch_file = PathJoinSubstitution(
        [
            pkg_my_robot_bringup,
            "launch",
            "rtabmap_mono_lidar.launch.py",  # YOUR RTAB-MAP SLAM LAUNCH
        ]
    )
    rviz_config_file = PathJoinSubstitution(
        [
            pkg_my_robot_bringup,
            "rviz",
            "rtabmap_lidar_only.rviz",  # Or your preferred Nav RViz
        ]
    )

    # --- 1. RTAB-Map SLAM Launch ---
    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(rtabmap_slam_launch_file),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "log_level": log_level,
            "use_rviz": "false",
        }.items(),
    )

    # --- 2. Local Costmap Node ---
    local_costmap_node = Node(
        package=pkg_custom_nav.split("/")[-1],
        executable="local_costmap_node",
        name="local_costmap_node",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "global_frame_for_costmap": "rtabmap_odom",
                "robot_base_frame": "base_link",
                "lidar_topic": "/scan",
                "width_meters": 3.5,
                "height_meters": 3.5,
                "resolution": 0.05,
                "costmap_update_frequency": 5.0,
                "obstacle_cost_value": 100,
                "transform_timeout_sec": 0.4,
                # Inflation parameters for local costmap
                "inflation_radius_m": 0.30,  # Max distance inflation spreads. TUNE THIS CAREFULLY!
                # Should be robot_radius + desired_clearance_from_obstacles.
                # Example: 0.2m robot radius + 0.15m clearance = 0.35m.
                "max_inflation_cost": 90,  # Max cost for an inflated cell (0-99, must be < 100).
                "cost_scaling_factor": 10.0,  # How sharply cost drops. Higher = faster drop. TUNE!
            }
        ],
    )

    # --- 3. Simple Global Planner Node ---
    simple_global_planner_node = Node(
        package=pkg_custom_nav.split("/")[-1],
        executable="simple_global_planner_node",
        name="simple_global_planner_node",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "goal_topic": "/goal_pose",
                "path_publish_topic": "/global_path",
                "map_topic": "/map",
                "robot_base_frame": "base_link",
                "global_frame": "map",
                "cost_lethal_threshold": 90,
                "cost_unknown_as_lethal": True,
                "cost_inflated_max": 220,  # Max cost for regular inflation (must be < TEMPORARY_AVOIDANCE_COST)
                "cost_neutral": 1,
                "enable_path_simplification": True,
                # Temporary avoidance zone parameters for global planner
                "temp_avoidance_topic": "/temp_avoidance_points",
                "temp_avoidance_point_lifetime_s": 30.0,  # How long to remember an avoidance point
                "astar_turn_penalty_cost": 7.0,  # INCREASED SIGNIFICANTLY
                "simplification_obstacle_check_expansion_cells": 1,  # Ensure it's 1 for safety brush
                "inflation_radius_m": 0.45,  # INCREASED (for static obstacles)
                "cost_penalty_multiplier": 2.5,  # INCREASED (to make A* respect inflation more)
                "temp_avoidance_radius_m": 0.60,  # INCREASED (A* outer inflation for temp)
                "simplification_max_allowed_cost": 50,  # DECREASED (LOS respects more static inflation)
                "simplification_temp_obstacle_clearance_radius_m": 0.35,  # INCREASED (LOS hard boundary for temp)
                "simplification_min_angle_change_deg": 5.0,  # Keep for final polish
            }
        ],
    )

    # --- 4. Tank Waypoint Navigator Node (blimp_goto_node) ---
    tank_navigator_node = Node(
        package=pkg_custom_nav.split("/")[-1],
        executable="blimp_goto_node",
        name="blimp_goto_node",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "max_linear_speed": 0.019,
                "min_approach_linear_speed": 0.013,
                "max_angular_speed_align": 0.18,
                "max_angular_speed_drive": 0.16,
                "min_active_angular_speed": 0.15,
                "kp_linear": 0.15,
                "kp_angular_align": 0.85,
                "kp_angular_drive": 0.35,
                "waypoint_position_tolerance": 0.08,
                "initial_alignment_tolerance": 0.10,
                "driving_keep_alignment_tolerance": 0.35,
                "lookahead_distance_drive": 0.4,
                "final_orientation_tolerance": 0.25,
                "angular_command_deadzone": 0.02,
                "robot_base_frame": "base_link",
                "map_frame": "map",
                "control_frequency": 10.0,
                "enable_final_orientation_at_last_waypoint": True,
                "pause_duration_after_turn": 0.3,
                "stuck_time_threshold": 4.0,
                "local_costmap_topic": "/local_costmap/costmap",
                "enable_obstacle_avoidance": True,
                "robot_footprint_length": 0.35,
                "robot_footprint_width": 0.20,
                "collision_check_time_horizon": 0.75,
                "num_trajectory_check_points": 5,
                "replan_if_path_blocked": True,
                "path_check_lookahead_distance_m": 1.0,
                "path_check_resolution_m": 0.1,
                "path_is_blocked_threshold": 60,  # Cost in local_costmap to consider path segment blocked
                "max_local_stuck_attempts": 1,
                "replan_request_timeout_s": 10.0,
                "single_goal_topic": "/goal_pose",
                "path_topic": "/global_path",
                "temp_avoidance_topic": "/temp_avoidance_points",  # For publishing blockage info
            }
        ],
    )

    # --- 5. RViz Node ---
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config_file],
        condition=IfCondition(use_rviz),
        parameters=[{"use_sim_time": use_sim_time}],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="false",
                description="Use simulation (Gazebo) clock if true",
            ),
            DeclareLaunchArgument(
                "log_level",
                default_value="info",
                description="Logging level for RTAB-Map",
            ),
            DeclareLaunchArgument(
                "use_rviz", default_value="true", description="Whether to start RViz"
            ),
            LogInfo(msg="Starting RTAB-Map SLAM components..."),
            slam_launch,
            LogInfo(msg="Starting Local Costmap Node... (delay 5s)"),
            TimerAction(period=5.0, actions=[local_costmap_node]),
            LogInfo(msg="Starting Simple Global Planner Node... (delay 5.5s)"),
            TimerAction(period=5.5, actions=[simple_global_planner_node]),
            LogInfo(msg="Starting Tank Waypoint Navigator... (delay 6s)"),
            TimerAction(period=6.0, actions=[tank_navigator_node]),
            LogInfo(msg="Starting RViz... (delay 7s)"),
            TimerAction(period=7.0, actions=[rviz_node]),
        ]
    )
