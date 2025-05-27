# ~/ros2_ws/src/my_robot_bringup/launch/libcamera_ros.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.actions import DeclareLaunchArgument
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    my_robot_bringup_share = get_package_share_directory('my_robot_bringup')

    camera_calibration_filename = 'freenove_8mp_pi5_640x480.yaml' # Placeholder
    camera_info_url = 'file://' + os.path.join(
        my_robot_bringup_share, 'config', 'camera_calibration', camera_calibration_filename
    )

    # Attempt to configure camera_ros for libcamera
    # Parameter names are speculative and depend on camera_ros implementation
    camera_params = [
        {'camera_name': 'freenove_camera'},
        {'frame_id': 'camera_link'}, # Must match your URDF
        {'camera_info_url': camera_info_url},
        {'image_width': 640},
        {'image_height': 480},
        {'framerate': 30.0},
        {'pixel_format': 'rgb8'}, # Desired output format from the node
        {'use_sim_time': use_sim_time},

        # Speculative parameters to try and select libcamera backend
        # These might be found in a params.yaml file for camera_ros or its documentation
        # Option 1: A general parameter for camera type/plugin
        # {'camera_type': 'libcamera'},
        # {'plugin_name': 'libcamera_driver::LibcameraDriver'}, # Example from other similar drivers

        # Option 2: If it loads a specific params file that then configures libcamera
        # You might need to find/create such a params file.
        # Example: PathJoinSubstitution([get_package_share_directory('camera_ros'), 'params', 'libcamera_params.yaml'])
    ]

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false', description='Use simulation clock if true'),

        Node(
            package='camera_ros',
            executable='camera_node', # Assuming this is the executable from 'ros2 pkg executables camera_ros'
            name='camera_node',
            namespace='camera',
            parameters=camera_params,
            output='screen'
        ),
    ])