import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, FindExecutable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import xacro

def generate_launch_description():
    pkg_my_robot_bringup = get_package_share_directory('my_robot_bringup')
    pkg_sllidar_ros2 = get_package_share_directory('sllidar_ros2')
    pkg_my_robot_description = get_package_share_directory('my_robot_description')

    # --- Launch Arguments ---
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    log_level = LaunchConfiguration('log_level', default='info')
    use_rviz = LaunchConfiguration('use_rviz', default='true')
    

    rtabmap_params_path = PathJoinSubstitution([
        pkg_my_robot_bringup, 'config', 'rtabmap', 'rtabmap_6dof_params.yaml'
    ])
    
    rviz_config_path = PathJoinSubstitution([
        pkg_my_robot_bringup, 'rviz', 'rtabmap_lidar_only.rviz' # Or your preferred one
    ])

    # --- Nodes ---

    # 1. Robot State Publisher (for TF from URDF)
    robot_description_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_my_robot_description, 'launch', 'rsp.launch.py'])
        ),
        launch_arguments={'use_sim_time': use_sim_time}.items()
    )

    # 2. SLLIDAR C1 Driver
    sllidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_sllidar_ros2, 'launch', 'sllidar_c1_launch.py'])
        ),
        launch_arguments={
            'serial_port': '/dev/ttyUSB0',
            'frame_id': 'lidar_link',
            'angle_compensate': 'true',
            'scan_mode': 'Standard',
            'use_sim_time': 'false'  # Use string instead of LaunchConfiguration
        }.items()
    )

    # 3. LaserScan to PointCloud2 Converter
    # This node converts each 2D LaserScan into a 3D PointCloud2,
    # using TF to place the points correctly in 3D space based on the lidar_link's pose.
    laserscan_to_pointcloud_node = Node(
        package='pointcloud_to_laserscan', # The package provides both conversion nodes
        executable='laserscan_to_pointcloud_node',
        name='laserscan_to_pointcloud_converter', # Unique name
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'target_frame': 'base_link',        # Transform points into the blimp's base_link frame.
                                                # This means each cloud represents the scan as seen from base_link.
                                                # Alternatively, use 'lidar_link' to get points in lidar's own frame,
                                                # then icp_odometry will use TF base_link -> lidar_link.
                                                # 'base_link' is often simpler for direct ICP consumption.
            'transform_tolerance': 0.1,         # Seconds
            'range_min': 0.12,                  # Min valid range for LiDAR points
            # 'min_height': -10.0,              # If using point_cloud_to_laserscan, not this node
            # 'max_height': 10.0,
            # 'angle_min': -3.14159,
            # 'angle_max': 3.14159,
            # 'angle_increment': 0.01745,       # These are more for PC to LaserScan
            # 'scan_time': 0.1,
            'inf_epsilon': 1.0,
            'use_inf': True,                    # Process +/- Inf range values if your LiDAR produces them
        }],
        remappings=[
            ('scan_in', '/scan'),               # Input LaserScan topic
            ('cloud', '/scan_cloud_converted')  # Output PointCloud2 topic
        ]
    )

    # 4. ICP Odometry Node (6DoF odometry from the converted PointCloud2)
    icp_odometry_node = Node(
        package='rtabmap_odom',
        executable='icp_odometry',
        name='icp_odometry',
        output='screen',
        parameters=[{
            'frame_id': 'base_link',            # TF frame of the robot base
            'odom_frame_id': 'odom',            # Frame for odometry output
            'publish_tf': True,                 # Publish odom -> base_link TF
            'use_sim_time': use_sim_time,
            'guess_frame_id': 'base_link',
            'wait_for_transform_duration': 0.2, # For base_link -> lidar_link
            
            'subscribe_scan_cloud': True,
            'approx_sync': True,
            'qos_scan_cloud': 2, # Reliable sensor_data

            # ICP Parameters (CRITICAL - TUNE THESE EXTENSIVELY)
            'Icp/Strategy': "1",                # 0=Minvariator, 1=LibPointMatcher
            'Icp/MaxCorrespondenceDistance': "0.5", # Start larger for blimp motion, then tune
            'Icp/PointToPlane': "true",
            'Icp/PointToPlaneK': "20",
            'Icp/VoxelSize': "0.05",            # Downsample cloud for ICP
            'Icp/Iterations': "20",
            'Icp/Epsilon': "0.001",
            'Icp/MaxRotation': "0.785",         # ~45 deg
            'Icp/MaxTranslation': "0.75",       # Allow larger movements for blimp
            'Icp/CorrespondenceRatio': "0.05",  # Min ratio of inliers, may need to be low

            'Odom/ScanKeyFrameThr': "0.4",
            'Odom/Strategy': "0",               # Frame-to-Map (local map of ICP)
            'Odom/GuessMotion': "true",
            'Odom/Deskewing': "false",          # Deskewing is harder with passive motion.
                                                # The LaserScanToPointCloud node transforming each scan
                                                # at its specific timestamp helps with this.
        }],
        remappings=[
            ('scan_cloud', '/scan_cloud_converted'), # Input from laserscan_to_pointcloud_node
            ('odom', '/odom')                        # Output odometry topic
        ]
    )

    # 5. RTAB-Map SLAM Node
    rtabmap_slam_node = Node(
        package='rtabmap_slam',
        executable='rtabmap',
        name='rtabmap',
        output='screen',
        arguments=['-d', '--ros-args', '--log-level', log_level], # '-d' deletes previous database on start
        parameters=[rtabmap_params_path, {
            'use_sim_time': use_sim_time,
            'qos_scan_cloud': 2, # Reliable
            'qos_odom': 2,       # Reliable
            'approx_sync': True,
        }],
        remappings=[
            ('scan_cloud', '/scan_cloud_converted'), # Input from laserscan_to_pointcloud_node
            ('odom', '/odom')                        # Input from icp_odometry
        ]
    )

    # 6. RViz Node
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

        LogInfo(msg="Launching Robot State Publisher..."),
        robot_description_launch,

        LogInfo(msg="Launching SLLidar..."),
        TimerAction(period=1.0, actions=[sllidar_launch]), # Give RSP time

        LogInfo(msg="Launching LaserScan to PointCloud2 Converter..."),
        TimerAction(period=2.0, actions=[laserscan_to_pointcloud_node]), # Needs /scan and TF

        LogInfo(msg="Launching ICP Odometry..."),
        TimerAction(period=3.0, actions=[icp_odometry_node]), # Needs /scan_cloud_converted

        LogInfo(msg="Launching RTAB-Map SLAM node..."),
        TimerAction(period=4.0, actions=[rtabmap_slam_node]), # Needs /scan_cloud_converted and /odom

        LogInfo(msg="Launching RViz (if enabled)..."),
        TimerAction(period=5.0, actions=[rviz_node]),
    ])