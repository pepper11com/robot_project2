import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.conditions import IfCondition

def generate_launch_description():
    # Package directories
    my_robot_bringup_dir = get_package_share_directory('my_robot_bringup')
    my_robot_description_dir = get_package_share_directory('my_robot_description')
    sllidar_ros2_dir = get_package_share_directory('sllidar_ros2')
    nav2_bringup_dir = get_package_share_directory('nav2_bringup') # For Nav2's own launch files

    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    map_yaml_file = LaunchConfiguration('map',
                                        default=os.path.join(
                                            my_robot_bringup_dir, 'maps', 'mymap.yaml')) # YOUR MAP
    params_file = LaunchConfiguration('params_file',
                                      default=os.path.join(
                                          my_robot_bringup_dir, 'config', 'nav2', 'my_robot_nav2_params.yaml')) # YOUR PARAMS
    autostart = LaunchConfiguration('autostart', default='true')
    use_rviz = LaunchConfiguration('use_rviz', default='true')
                                       # YOUR RVIZ CONFIG
    rviz_config = LaunchConfiguration('rviz_config',
                                       default=os.path.join(nav2_bringup_dir, 'rviz', 'nav2_default_view.rviz')) # USE NAV2'S DEFAULT

    # 1. Robot URDF and Robot State Publisher
    robot_description_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(my_robot_description_dir, 'launch', 'rsp.launch.py')
        ),
        launch_arguments={'use_sim_time': use_sim_time}.items()
    )

    # 2. Lidar Driver (SLLidar C1)
    sllidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(sllidar_ros2_dir, 'launch', 'sllidar_c1_launch.py')
        ),
        launch_arguments={
            'serial_port': '/dev/ttyUSB0',
            'frame_id': 'lidar_link', # Matches URDF
            'angle_compensate': 'true',
            'scan_mode': 'Standard',
            'use_sim_time': use_sim_time
        }.items()
    )

    # 3. ICP Odometry Node
    # Ensure parameters here are for publishing TF: "odom" -> "base_link" & msgs to /odom
    icp_odometry_node = Node(
        package='rtabmap_odom',
        executable='icp_odometry',
        name='icp_odometry',
        output='screen',
        parameters=[{
            'frame_id': 'base_link',
            'odom_frame_id': 'odom',        # *** NAV2 STANDARD ***
            'publish_tf': True,
            'use_sim_time': use_sim_time,
            'wait_for_transform': 0.3,      # Increased slightly
            'Odom/GuessMotion': "True",
            'Odom/Deskewing': "True",
            'Icp/Strategy': "1",
            'Icp/MaxCorrespondenceDistance': "0.3",
            'Icp/CorrespondenceRatio': "0.05",
            'Icp/MaxTranslation': "0.75",
            'Icp/MaxRotation': "1.57",
            'Icp/Iterations': "20",
            'scan_voxel_size': 0.05,
            'Icp/PointToPlane': "True",
            'scan_normal_k': 10,
            'Icp/Epsilon': "0.0001",
            'scan_range_min': 0.12,
            'scan_range_max': 10.0,
        }],
        remappings=[
            ('scan', '/scan'),
            ('odom', '/odom')              # *** NAV2 STANDARD ***
        ]
    )

    # 4. Nav2 Stack
    # We use nav2_bringup's navigation_launch.py as a base for the core Nav2 nodes
    # This is generally preferred over listing all Nav2 nodes manually.
    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(nav2_bringup_dir, 'launch', 'bringup_launch.py')),
        launch_arguments={
            'namespace': '', # No namespace
            'use_sim_time': use_sim_time,
            'autostart': autostart,
            'params_file': params_file, # YOUR custom Nav2 params
            'map': map_yaml_file, # YOUR map
            'use_composition': 'True',  # Usually recommended
            'use_respawn': 'False',     # Can be True for more robustness
            # 'container_name': 'nav2_container' # If use_composition is True
        }.items()
    )

    # 5. RViz
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(use_rviz)
    )

    return LaunchDescription([
    
        DeclareLaunchArgument('use_sim_time', default_value='false', description='Use simulation clock if true'),
        DeclareLaunchArgument('map', default_value=os.path.join(my_robot_bringup_dir, 'maps', 'mymap.yaml'), description='Full path to map file'),
        DeclareLaunchArgument('params_file', default_value=os.path.join(my_robot_bringup_dir, 'config', 'nav2', 'my_robot_nav2_params.yaml'), description='Full path to Nav2 params file'),
        DeclareLaunchArgument('autostart', default_value='true', description='Automatically startup Nav2 stack'),
        DeclareLaunchArgument('use_rviz', default_value='true', description='Launch RViz'),
        DeclareLaunchArgument('rviz_config', default_value=os.path.join(nav2_bringup_dir, 'rviz', 'nav2_default_view.rviz'), description='Full path to RViz config file'),

        LogInfo(msg="Launching Robot Description (URDF & Robot State Publisher)..."),
        robot_description_launch,

        LogInfo(msg="Launching SLLidar Driver..."),
        sllidar_launch,
        
        LogInfo(msg="Launching ICP Odometry Node..."),
        icp_odometry_node,

        LogInfo(msg="Launching Nav2 Stack..."),
        nav2_launch,

        LogInfo(msg="Launching RViz..."),
        rviz_node,
    ])