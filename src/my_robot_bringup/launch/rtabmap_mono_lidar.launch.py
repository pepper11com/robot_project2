import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_my_robot_bringup = get_package_share_directory('my_robot_bringup')
    pkg_my_robot_description = get_package_share_directory('my_robot_description')
    pkg_sllidar_ros2 = get_package_share_directory('sllidar_ros2')

    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    log_level = LaunchConfiguration('log_level', default='info')
    use_rviz = LaunchConfiguration('use_rviz', default='true')
    # Camera is disabled for this setup
    launch_d405_camera = LaunchConfiguration('launch_d405_camera', default='false')

    # POINT TO THE ROBUST LIDAR-ONLY YAML
    rtabmap_config_path = PathJoinSubstitution([
        pkg_my_robot_bringup, 'config', 'rtabmap', 'rtabmap_lidar_only_icp.yaml'
    ])

    rviz_config_path = PathJoinSubstitution([
        pkg_my_robot_bringup, 'rviz', 'rtabmap_lidar_only.rviz' # Or your preferred one
    ])

    robot_description_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_my_robot_description, 'launch', 'rsp.launch.py'])
        ),
        launch_arguments={'use_sim_time': use_sim_time}.items()
    )

    # Camera launch is skipped due to launch_d405_camera being false by default

    sllidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_sllidar_ros2, 'launch', 'sllidar_c1_launch.py'])
        ),
        launch_arguments={
            'serial_port': '/dev/ttyUSB0',
            'frame_id': 'lidar_link',
            'angle_compensate': 'true',
            'scan_mode': 'Standard',
            'use_sim_time': use_sim_time
        }.items()
    )

    icp_odometry_node = Node(
        package='rtabmap_odom',
        executable='icp_odometry',
        name='icp_odometry',
        output='screen',
        parameters=[{
            'frame_id': 'base_link',
            'odom_frame_id': 'rtabmap_odom',
            'publish_tf': True,
            'use_sim_time': use_sim_time,
            'wait_for_transform': 0.2, # How long to wait for TF lookups (e.g. base_link -> lidar_link)

            # General Odometry Strategy
            'Odom/Strategy': "0",             # 0 for Frame-to-Map (ICP Odometry uses this concept for its local map)
            'Odom/GuessMotion': "True",
            'Odom/Deskewing': "True",         # Enable if lidar provides timestamps per point

            # ICP Algorithm specific to rtabmap_odom
            'Icp/Strategy': "1",              # Use libpointmatcher
            'Icp/MaxCorrespondenceDistance': "0.3", # Increased from default 0.1 to be more tolerant
            'Icp/CorrespondenceRatio': "0.05",  # Lowered from default 0.1, accepts transform with fewer matches (more tolerant for fast turns/low overlap)
            'Icp/MaxTranslation': "0.75",     # Increased from 0.5, allows larger translational jumps if needed
            'Icp/MaxRotation': "1.57",        # ~90 degrees, increased from 1.0 for very fast point turns
            'Icp/Iterations': "20",           # Reduced from 30, might speed up if convergence is quick
            'Icp/VoxelSize': "0.05",            # Passed as scan_voxel_size to node, but Icp/VoxelSize is the internal
            'scan_voxel_size': 0.05,          # This sets Icp/VoxelSize for rtabmap_odom node
            'Icp/PointToPlane': "True",         # Generally more accurate
            'scan_normal_k': 10,              # K neighbors for normal estimation (passed to Icp/PointToPlaneK), increased from 5
            # 'Icp/OutlierRatio': "0.7",      # For libpointmatcher, default 0.85. Lower = more outlier rejection.
            'Icp/Epsilon': "0.0001",          # Stricter convergence criterion

            # Scan processing parameters for ICP Odometry
            'scan_range_min': 0.12,
            'scan_range_max': 10.0,
            # 'Odom/ScanKeyFrameThr': "0.7",  # Default 0.9. Lower means new "key scan" for ICP local map more often. Could help with fast changes.
        }],
        remappings=[
            ('scan', '/scan'),
            ('odom', '/rtabmap_odom_raw_msgs')
        ]
    )

    rtabmap_slam_node = Node(
        package='rtabmap_slam',
        executable='rtabmap',
        name='rtabmap',
        output='screen',
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
        parameters=[rtabmap_config_path, {
            'subscribe_stereo': False,
            'subscribe_scan': True,
            'odom_frame_id': 'rtabmap_odom',
            'map_frame_id': 'map',
            'use_sim_time': use_sim_time,
            'qos_scan': 1, # Reliable
            'qos_odom': 1, # Reliable

            # --- CRITICAL CHANGES FOR LOCALIZATION MODE ---
            'delete_db_on_start': False,    # Ensure this is false
            'Mem/IncrementalMemory': 'False', # <<< SET TO FALSE FOR LOCALIZATION
            'Mem/InitWMWithAllNodes': 'True', # <<< SET TO TRUE TO LOAD ENTIRE MAP FROM DB for localization
        }],
        remappings=[
            ('scan', '/scan'),
            ('odom', '/rtabmap_odom_raw_msgs'),
        ]
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_path],
        condition=IfCondition(use_rviz),
        parameters=[{'use_sim_time': use_sim_time}]
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('log_level', default_value='info'),
        DeclareLaunchArgument('use_rviz', default_value='true'),
        DeclareLaunchArgument('launch_d405_camera', default_value='false'),

        LogInfo(msg="Launching Robot Description..."),
        robot_description_launch,

        LogInfo(msg="Launching SLLidar..."),
        TimerAction(period=1.0, actions=[sllidar_launch]),

        LogInfo(msg="Launching ICP Odometry (Robust Lidar)..."),
        TimerAction(period=2.5, actions=[icp_odometry_node]),

        LogInfo(msg="Launching RTAB-Map SLAM (LIDAR ONLY - Robust ICP)..."),
        TimerAction(period=4.0, actions=[rtabmap_slam_node]),

        LogInfo(msg="Launching RViz (if enabled)..."),
        TimerAction(period=4.5, actions=[rviz_node]),
    ])