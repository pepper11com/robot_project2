import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    
    my_robot_bringup_dir = get_package_share_directory('my_robot_bringup')
    cartographer_config_dir = os.path.join(my_robot_bringup_dir, 'config', 'cartographer') # Corrected path
    configuration_basename = 'c1_lidar_2d.lua' # Corrected path
    
    # This default_map_pbstream will be used if load_state_filename is not overridden at launch time
    default_map_pbstream = os.path.join(my_robot_bringup_dir, 'maps', 'my_map.pbstream') # Use your actual map name
    load_state_filename = LaunchConfiguration('load_state_filename', default=default_map_pbstream)
    
    # Get robot description package directory
    my_robot_description_pkg_dir = get_package_share_directory('my_robot_description')
    
    # Path to the robot description launch file
    robot_description_launch_file = PathJoinSubstitution([
        my_robot_description_pkg_dir, 'launch', 'view_model.launch.py' # Assuming this is the correct name
    ])
    
    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation (Gazebo) clock if true'
        ),
        
        DeclareLaunchArgument(
            'load_state_filename',
            default_value=default_map_pbstream, # Points to your map
            description='Full path to the .pbstream map file to load'
        ),
        
        # Include the robot_state_publisher launch file
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(robot_description_launch_file),
            launch_arguments={'use_sim_time': use_sim_time}.items() # Pass use_sim_time if description.launch.py uses it
        ),
        
        Node(
            package='cartographer_ros',
            executable='cartographer_node',
            name='cartographer_node',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}],
            arguments=[
                '-configuration_directory', cartographer_config_dir,
                '-configuration_basename', configuration_basename,
                '-load_state_filename', load_state_filename, # Loads the old map
                # NO -start_trajectory_with_default_topics=false, so it WILL start a new trajectory
                '-initial_pose_estimate_frozen', 'false', # Allows it to search for initial pose
            ],
            remappings=[
                ('scan', '/scan')
            ]
        ),
        
        Node(
            package='cartographer_ros',
            executable='cartographer_occupancy_grid_node',
            name='occupancy_grid_node',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}],
            arguments=['-resolution', '0.05', '-publish_period_sec', '1.0']
        ),
    ])