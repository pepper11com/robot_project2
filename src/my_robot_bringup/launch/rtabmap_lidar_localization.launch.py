#!/usr/bin/env python3
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Launch arguments
    db_path_arg = DeclareLaunchArgument(
        'database_path',
        default_value= '',
        description='Full path to RTAB-Map database (.db) file to load'
    )
    sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo) clock if true'
    )

    database_path = LaunchConfiguration('database_path')
    use_sim_time  = LaunchConfiguration('use_sim_time')

    # Optional: point this to your RViz config under my_robot_bringup/rviz
    rviz_config = PathJoinSubstitution(
        [FindPackageShare('my_robot_bringup'), 'rviz', 'localization.rviz']
    )

    return LaunchDescription([
        db_path_arg,
        sim_time_arg,

        # RViz
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config],
            parameters=[{'use_sim_time': use_sim_time}]
        ),

        # ICP Odometry
        Node(
            package='rtabmap_odom',
            executable='icp_odometry',
            name='icp_odometry',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}],
            remappings=[
                ('/scan', '/scan'),
                ('/scan_cloud', '/scan_cloud')
            ]
        ),

        # RTAB-Map in localization mode
        Node(
            package='rtabmap_slam',
            executable='rtabmap',
            name='rtabmap',
            output='screen',
            parameters=[
                {'use_sim_time': use_sim_time},
                {'database_path': database_path},
                # load-only (no new mapping)
                {'Mem/IncrementalMemory': False},
                {'Mem/InitWMWithAllNodes': True},
                # frames
                {'frame_id': 'base_link'},
                {'odom_frame_id': 'rtabmap/odom'},
                {'map_frame_id': 'map'},
                # subscribe only LIDAR + odom info
                {'subscribe_scan':      True},
                {'subscribe_scan_cloud': False},
                {'subscribe_odom_info': True},
                # publish only the map, not images
                {'Publish/Map':      True},
                {'Publish/Odom':     False},
                # ensure a continuous, stable map topic
                {'Map/UpdateGraph':   False},
                {'Map/PublishRate':   1.0},
                {'Map/AlwaysUpdate':  True},
            ],
            remappings=[
                ('scan',             '/scan'),
                ('scan_cloud',       '/scan_cloud'),
                ('odom_info',        '/odom_info'),
            ]
        ),

        # Assemble and re-publish the occupancy grid / map topic
        Node(
            package='map_assembler',
            executable='map_assembler',
            name='map_assembler',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}],
            remappings=[
                # listen to RTAB-Map’s map data topic
                ('data', 'rtabmap/mapData'),
                # republish as standard /map for nav2/RViz
                ('map',  'map')
            ]
        ),
    ])
