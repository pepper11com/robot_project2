import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, TimerAction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.conditions import IfCondition 

def generate_launch_description():
    # --- Define Package Shares ---
    # Adjust these to your actual package names if different
    pkg_my_robot_bringup = get_package_share_directory('my_robot_bringup')
    # Assuming your custom navigation scripts (tank_waypoint_navigator.py, local_costmap_node.py, rviz_path_collector.py)
    # are in a package named 'my_robot_navigation' in its 'scripts' directory.
    # If they are in 'my_robot_bringup/scripts', change this accordingly.
    pkg_custom_nav = get_package_share_directory('my_robot_driver') # CHANGE IF YOUR SCRIPTS ARE IN A DIFFERENT PKG

    # --- Launch Arguments ---
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    log_level = LaunchConfiguration('log_level', default='info') # For RTAB-Map
    use_rviz = LaunchConfiguration('use_rviz', default='true')
    # Path to your RTAB-Map SLAM launch file (the one you use for mapping)
    # This launch file should start your lidar, robot_state_publisher, icp_odometry, and rtabmap.
    rtabmap_slam_launch_file = PathJoinSubstitution([
        pkg_my_robot_bringup, 'launch', 'rtabmap_mono_lidar.launch.py' # YOUR RTAB-MAP SLAM LAUNCH
    ])
    # RViz configuration file (optional, can use a default one)
    rviz_config_file = PathJoinSubstitution([
        pkg_my_robot_bringup, 'rviz', 'rtabmap_lidar_only.rviz' # Or your preferred Nav RViz
    ])

    # --- 1. Include your RTAB-Map SLAM Launch ---
    # This will bring up lidar, robot_state_publisher, icp_odometry, and rtabmap (in mapping or localization mode)
    # Ensure your rtabmap_mapping.launch.py uses 'rtabmap_odom' as odom_frame_id for icp_odometry
    # and rtabmap_slam_node subscribes to /rtabmap_odom_raw_msgs (or whatever ICP publishes)
    # and that TF base_link -> rtabmap_odom -> map is established.
    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(rtabmap_slam_launch_file),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'log_level': log_level,
            'use_rviz': 'false' # We will launch our own RViz instance later
        }.items()
    )

    # --- 2. Local Costmap Node ---
    local_costmap_node = Node(
        package=pkg_custom_nav.split('/')[-1], # Extracts package name if scripts are in this package
        # If using entry points from setup.py, use executable='local_costmap'
        executable='local_costmap_node', # Assumes it's in pkg_custom_nav/scripts/
        name='local_costmap_node',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'global_frame_for_costmap': 'rtabmap_odom', # IMPORTANT: Frame for local costmap origin
            'robot_base_frame': 'base_link',
            'lidar_topic': '/scan',
            'width_meters': 3.5,        # Size of the rolling window costmap
            'height_meters': 3.5,
            'resolution': 0.05,         # meters/cell
            'inflation_radius_m': 0.10, # Robot radius + safety buffer (TUNE THIS)
            'costmap_update_frequency': 5.0, # Hz
            'obstacle_cost_value': 100,
            'transform_timeout_sec': 0.2
        }],
        # If scripts are in pkg_custom_nav/scripts, and pkg_custom_nav is 'my_robot_bringup'
        # you might need to specify the full path if not using entry_points and colcon doesn't find it.
        # cwd=PathJoinSubstitution([pkg_custom_nav, 'scripts']) # Or ensure scripts are in PATH
    )
    
    
    simple_global_planner_node = Node(
        package=pkg_custom_nav.split('/')[-1], # Extracts package name if scripts are in this package
        executable='simple_global_planner_node', # or entry point
        name='simple_global_planner',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'goal_topic': '/goal_pose',
            'path_publish_topic': '/global_path',
            'map_topic': '/map',
            'robot_base_frame': 'base_link',
            'global_frame': 'map',
            
            
            'inflation_radius_m': 0.4, # Adjust based on robot size + safety
            'cost_lethal_threshold': 90,
            'cost_unknown_as_lethal': True,
            'cost_inflated_max': 200, # Max cost for inflated cells (less than lethal)
            'cost_neutral': 1,        # Cost of free space
            'cost_penalty_multiplier': 4.0, # How much to penalize moving through inflated (higher = avoid more)
            
            # NEW PARAMETERS FOR PATH SIMPLIFICATION
            'enable_path_simplification': True,
            'simplification_obstacle_check_expansion_cells': 2, # 0 or 1 is usually good.
                                                              # 0 = check only direct line cells
                                                              # 1 = check direct line + 1 cell around each
                                                              
            'simplification_max_allowed_cost': 50 # START HERE. Range: cost_neutral < X < cost_inflated_max
        }]
    )

    # --- 3. Tank Waypoint Navigator Node ---
    tank_navigator_node = Node(
        package=pkg_custom_nav.split('/')[-1],
        executable='blimp_goto_node',
        name='blimp_goto_node',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'max_linear_speed': 0.014,       # TUNE
            'min_approach_linear_speed': 0.015, # TUNE
            'max_angular_speed_align': 0.18,  # TUNE
            'max_angular_speed_drive': 0.06, # TUNE
            'kp_linear': 0.15,               # TUNE
            'kp_angular_align': 0.8,         # TUNE
            'kp_angular_drive': 0.35,        # TUNE
            'waypoint_position_tolerance': 0.08, # TUNE
            'initial_alignment_tolerance': 0.10, # TUNE (radians)
            'driving_keep_alignment_tolerance': 0.35, # TUNE (radians)
            'lookahead_distance_drive': 0.5,       # Meters
            'final_orientation_tolerance': 0.25,    # TUNE (radians)
            'robot_base_frame': 'base_link',
            'map_frame': 'map', # Global frame for waypoints/goals
            'local_costmap_topic': '/local_costmap/costmap',
            'enable_obstacle_avoidance': False, # Set to True if you want obstacle avoidance
            'robot_footprint_length': 0.35,  # TUNE: Actual robot length
            'robot_footprint_width': 0.20,   # TUNE: Actual robot width
            'collision_check_time_horizon': 0.8, # TUNE
            'num_trajectory_check_points': 7,    # TUNE
            'obstacle_avoidance_turn_speed_factor': 0.6, # TUNE
            'obstacle_avoidance_duration_s': 3.0, # TUNE
            'path_topic': '/global_path',
            'single_goal_topic': '/goal_pose', # For single goals if you use it
            
            'pause_duration_after_turn': 0.3, # Seconds
        }]
    )

    # --- 5. RViz Node ---
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_file],
        condition=IfCondition(use_rviz),
        parameters=[{'use_sim_time': use_sim_time}]
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false', description='Use simulation (Gazebo) clock if true'),
        DeclareLaunchArgument('log_level', default_value='info', description='Logging level for RTAB-Map'),
        DeclareLaunchArgument('use_rviz', default_value='true', description='Whether to start RViz'),

        LogInfo(msg="Starting RTAB-Map SLAM components..."),
        slam_launch, # Starts lidar, robot_state_pub, icp_odom, rtabmap

        LogInfo(msg="Starting Local Costmap Node..."),
        TimerAction(period=5.0, actions=[local_costmap_node]), # Needs TF map->odom->base_link and /scan

        LogInfo(msg="Starting Simple Global Planner Node..."),
        TimerAction(period=5.5, actions=[simple_global_planner_node]), # Needs map and TF
        
        LogInfo(msg="Starting Tank Waypoint Navigator..."),
        TimerAction(period=6.0, actions=[tank_navigator_node]), # Needs TF and local_costmap

        LogInfo(msg="Starting RViz..."),
        TimerAction(period=7.0, actions=[rviz_node]),
    ])