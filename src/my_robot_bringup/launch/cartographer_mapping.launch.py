import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    
    my_robot_bringup_dir = get_package_share_directory('my_robot_bringup')
    cartographer_config_dir = os.path.join(my_robot_bringup_dir, 'config', 'cartographer')
    configuration_basename = 'c1_lidar_2d.lua'
    
    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation (Gazebo) clock if true'
        ),
        
        
        Node(
            package='cartographer_ros',
            executable='cartographer_node', # This one is likely correct as 'cartographer_node'
            name='cartographer_node',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}],
            arguments=['-configuration_directory', cartographer_config_dir,
                       '-configuration_basename', configuration_basename],
            remappings=[
                ('scan', '/scan')
            ]
        ),
        
        Node(
            package='cartographer_ros',
            executable='cartographer_occupancy_grid_node', # <--- CORRECTED HERE
            name='occupancy_grid_node', # The 'name' can remain as you like
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}],
            arguments=['-resolution', '0.05', '-publish_period_sec', '1.0']
        ),
    ])