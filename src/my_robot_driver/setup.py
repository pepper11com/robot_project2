from setuptools import find_packages, setup
import glob
import os

package_name = 'my_robot_driver'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include all launch files
        (os.path.join('share', package_name, 'launch'),
            glob.glob('launch/*.launch.py')),
        # Include all YAML files in config directory
        (os.path.join('share', package_name, 'config'),
            glob.glob('config/*.yaml')),
        
    ],
    install_requires=['setuptools', 'gpiozero', 'rclpy', 'geometry_msgs'],
    zip_safe=True,
    maintainer='pepper11com',
    maintainer_email='pepper11com@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'motor_controller_node = my_robot_driver.motor_controller_node:main',
            'blimp_goto_node = my_robot_driver.blimp_goto_node:main',
            'rviz_path_collector_node = my_robot_driver.rviz_path_collector:main',
            'local_costmap_node = my_robot_driver.local_costmap_node:main',
            'simple_global_planner_node = my_robot_driver.simple_global_planner_node:main',
        ],
    },
)

# cd ~/ros2_ws
# rm -rf build install log
# cd ~/ros2_ws
# Make sure your ROS 2 environment is sourced
# source /opt/ros/jazzy/setup.bash 
# colcon build

# ros2 run teleop_twist_keyboard teleop_twist_keyboard
# ros2 run my_robot_driver motor_controller_node
# source /home/pepper11com/py_envs/picamera_env/bin/activate