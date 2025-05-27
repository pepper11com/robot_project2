# ~/ros2_ws/src/my_robot_bringup/launch/rtabmap_localization_final_attempt.launch.py
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    database_path_arg = LaunchConfiguration('database_path', default='~/.ros/rtabmap.db')

    bringup_share = get_package_share_directory('my_robot_bringup')
    
    # CRITICAL: Point to your ORIGINAL, WORKING SLAM YAML configuration.
    # Assuming it was named 'rtabmap_lidar_only.yaml'
    rtabmap_base_config_file_path = PathJoinSubstitution([
        bringup_share, 'config', 'rtabmap', 'rtabmap_lidar_only.yaml'
    ])

    robot_description_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                get_package_share_directory('my_robot_description'),
                'launch', 'description.launch.py' # Adjust if your description launch file has a different name/path
            ])
        ),
        launch_arguments={'use_sim_time': use_sim_time}.items()
    )

    static_tf_odom_to_rtabmap_odom = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='odom_to_rtabmap_odom_static_tf', # Slightly more descriptive name
        arguments=['0', '0', '0', '0', '0', '0', 'odom', 'rtabmap/odom'],
        parameters=[{'use_sim_time': use_sim_time}]
    )

    icp_odometry_node = Node(
        package='rtabmap_odom',
        executable='icp_odometry',
        name='icp_odometry',
        output='screen',
        parameters=[{
            'frame_id': 'base_link',
            'odom_frame_id': 'rtabmap/odom',
            'publish_tf': True,
            'scan_topic': '/scan',
            'qos_scan': 1,
            'scan_range_min': 0.12,
            'scan_range_max': 10.0,
            'scan_voxel_size': 0.05,
            'scan_normal_k': 5,
            'use_sim_time': use_sim_time,
        }],
        remappings=[
            ('scan', '/scan'),
            ('odom', '/rtabmap/odom'),
        ]
    )

    rtabmap_localization_node = Node(
        package='rtabmap_slam', # The package name is 'rtabmap_slam' for the 'rtabmap' executable
        executable='rtabmap',
        name='rtabmap', # Keeping the node name as 'rtabmap'
        output='screen',
        # NO --delete_db_on_start argument for localization
        parameters=[
            rtabmap_base_config_file_path, # Load base configuration from the original SLAM YAML
            {
                # --- Core RTAB-Map Parameters for Localization Mode ---
                'use_sim_time': use_sim_time,
                'database_path': database_path_arg,
                
                # These parameters are critical for localization-only mode
                # Using strings to ensure override if RTAB-Map's parser expects them this way from launch.
                "Mem/IncrementalMemory": "false",       # Disable adding new data to map graph
                "Mem/InitWMWithAllNodes": "true",       # Load all map nodes from DB into working memory
                
                "map_always_update": False,           # Disable global map updates
                "map_empty_ray_tracing": True,        # Keep for visualizing grid map

                "RGBD/AngularUpdate": "0.0",            # Don't create new map nodes on angular motion
                "RGBD/LinearUpdate": "0.0",             # Don't create new map nodes on linear motion

                "Rtabmap/StartNewMapOnLoopClosure": "false", # Don't start a new map if relocalized

                # --- Grid Map Parameters (to prevent Grid/RangeMax being set to 0) ---
                # These override any settings from the YAML or automatic adjustments
                "Grid/Sensor": "0",                     # Explicitly use LaserScan (0) for grid
                "Grid/FromDepth": "false",              # Confirm grid is not from depth projection
                "Grid/RangeMax": "10.0",                # Force this value (from your original YAML)
                "Grid/CellSize": "0.05",                # Force this value (from your original YAML)
                "Grid/RangeMin": "0.12",                # Force this value (from your original YAML)
                "Grid/MapNegativeScansEmptyOverRayTrace": "true" # from your original YAML
            }
        ],
        remappings=[
            ('scan', '/scan'),
            ('odom', '/rtabmap/odom'),
            # ('mapData', '/rtabmap/mapData'), # For rtabmap_viz or debugging
            # ('grid_map', '/rtabmap/grid_map') # Default topic for rtabmap's grid
        ]
        # arguments=['-d'] # Uncomment for rtabmap core debug messages
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time', default_value='false',
            description='Use simulation (Gazebo) clock if true'
        ),
        DeclareLaunchArgument(
            'database_path', default_value='~/.ros/rtabmap.db',
            description='Path to the RTAB-Map database file for localization.'
        ),
        LogInfo(msg=["--- Launching RTAB-Map in Localization Mode (Final Attempt Configuration) ---"]),
        LogInfo(msg=["Using base RTAB-Map config from: ", rtabmap_base_config_file_path]),
        LogInfo(msg=["Attempting to load database from: ", database_path_arg]),
        LogInfo(msg=["Ensure your original SLAM YAML ('rtabmap_lidar_only.yaml') is correct and in place."]),

        robot_description_launch,
        static_tf_odom_to_rtabmap_odom,
        icp_odometry_node,
        rtabmap_localization_node,
    ])