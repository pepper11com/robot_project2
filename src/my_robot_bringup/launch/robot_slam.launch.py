import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    
    my_robot_bringup_pkg_dir = get_package_share_directory('my_robot_bringup')
    my_robot_description_pkg_dir = get_package_share_directory('my_robot_description')

    # Path to Cartographer mapping launch file
    cartographer_mapping_launch_file = PathJoinSubstitution([
        my_robot_bringup_pkg_dir, 'launch', 'cartographer_mapping.launch.py'
    ])

    # Path to robot description launch file
    robot_description_launch_file = PathJoinSubstitution([
        my_robot_description_pkg_dir, 'launch', 'description.launch.py' # Assuming this is the correct name
    ])

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation (Gazebo) clock if true'
        ),

        # Include the robot_state_publisher launch file
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(robot_description_launch_file),
            launch_arguments={'use_sim_time': use_sim_time}.items() # Pass use_sim_time if description.launch.py uses it
        ),

        # Include the Cartographer mapping launch file
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(cartographer_mapping_launch_file),
            launch_arguments={'use_sim_time': use_sim_time}.items()
        ),
        
        # You might also include your sllidar_ros2 launch here if you want a single launch file
        # Example:
        # IncludeLaunchDescription(
        #     PythonLaunchDescriptionSource([
        #         get_package_share_directory('sllidar_ros2'), '/launch/sllidar_launch.py' # Check exact path
        #     ])
        # ),
    ])