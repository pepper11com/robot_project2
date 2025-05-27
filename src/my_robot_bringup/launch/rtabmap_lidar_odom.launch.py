import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Allow using simulation time
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')

    # Path to our RTAB-Map YAML config
    bringup_share = get_package_share_directory('my_robot_bringup')
    rtabmap_cfg = PathJoinSubstitution([
        bringup_share, 'config', 'rtabmap', 'rtabmap_lidar_only.yaml'
    ])

    # 1) Robot description (URDF + TF)
    robot_desc = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                get_package_share_directory('my_robot_description'),
                'launch', 'description.launch.py'
            ])
        ),
        launch_arguments={'use_sim_time': use_sim_time}.items()
    )

    # 2) Static transform: odom → rtabmap/odom
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='odom_to_rtabmap_odom',
        arguments=['0', '0', '0', '0', '0', '0', 'odom', 'rtabmap/odom']
    )

    # 3) ICP Odometry node: publishes /rtabmap/odom + TF
    icp_odometry = Node(
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
            # tuning parameters
            'scan_range_min': 0.12,
            'scan_range_max': 10.0,
            'scan_voxel_size': 0.05,
            'scan_normal_k': 5,
            
            # Add these parameters for better handling of sharp turns
            'wait_for_transform': 0.2,     # Wait longer for transforms
            'wait_icp_transform': 0.2,     # Wait more during ICP
            'icp_max_rotational_velocity': 1.0,  # Limit max rotation rate
            'icp_max_translation': 0.5,   # Limit max translation
            'icp_max_correspondence_distance': 0.3,  # More tolerant matching
            'icp_iterations': 40,          # More iterations for convergence
        }],
        remappings=[
            ('scan', '/scan'),
            ('odom', '/rtabmap/odom'),
        ]
    )

    # 4) RTAB-Map SLAM node: consumes ICP odom + /scan, builds 2D map
    rtabmap_slam = Node(
        package='rtabmap_slam',
        executable='rtabmap',
        name='rtabmap',
        output='screen',
        arguments=['--delete_db_on_start'],
        parameters=[rtabmap_cfg, {
            'use_sim_time': use_sim_time,
        }],
        remappings=[
            ('scan', '/scan'),
            ('odom', '/rtabmap/odom'),
        ]
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time', default_value='false',
            description='Use simulation (Gazebo) clock'
        ),
        robot_desc,
        static_tf,
        icp_odometry,
        rtabmap_slam,
    ])
