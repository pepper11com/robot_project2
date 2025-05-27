import os
from glob import glob
from setuptools import setup # find_packages is not needed if 'packages' is explicit

package_name = 'my_robot_bringup'

setup(
    name=package_name,
    version='0.0.1', # Match package.xml
    packages=[package_name], # This expects your python modules to be in a 'my_robot_bringup/my_robot_bringup/' subdirectory
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        
        # Install launch files from the 'launch' directory
        (os.path.join('share', package_name, 'launch'), 
            glob(os.path.join('launch', '*.launch.py'))),
        
        # Install Cartographer Lua configuration files
        # Source: my_robot_bringup/config/cartographer/c1_lidar_2d.lua
        # Destination: install/my_robot_bringup/share/my_robot_bringup/config/cartographer/c1_lidar_2d.lua
        (os.path.join('share', package_name, 'config', 'cartographer'), 
            glob(os.path.join('config', 'cartographer', '*.lua'))),
        
        # If you have other config files directly under 'config/' (e.g., my_robot_bringup/config/params.yaml)
        # and NOT in subdirectories like 'config/cartographer', you would add:
        (os.path.join('share', package_name, 'config'), 
             glob(os.path.join('config', '*.yaml'))), # Example for *.yaml files in config/
        
         (os.path.join('share', package_name, 'config'), 
             glob(os.path.join('config', '*.lua'))), # Example for *.lua files in config/

        # Install maps directory (creates an empty one if 'maps/' source dir is empty, copies contents if any)
        (os.path.join('share', package_name, 'maps'), 
            glob(os.path.join('maps', '*'))), 
        
        (os.path.join('share', package_name, 'config', 'rtabmap'),
            glob(os.path.join('config', 'rtabmap', '*.yaml'))),
        
        (os.path.join('share', package_name, 'config', 'nav2'),
            glob(os.path.join('config', 'nav2', '*.yaml'))),
        
        # Install scripts from the 'scripts' directory
        # Destination: install/my_robot_bringup/lib/my_robot_bringup/
        (os.path.join('lib', package_name), [
            os.path.join('scripts', 'odom_tf_broadcaster'), # Assuming this is a script
            os.path.join('scripts', 'rf2o_qos_wrapper.py')  # Assuming this is a script
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='pepper11com',
    maintainer_email='pepper11com@example.com', # Update email if needed
    description='Robot bringup package with Cartographer SLAM', # Match package.xml
    license='Apache-2.0', # Match package.xml
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # If 'odom_tf_broadcaster' or 'rf2o_qos_wrapper.py' are Python nodes,
            # they should be structured as modules within the 'my_robot_bringup' Python package
            # (e.g., my_robot_bringup/my_robot_bringup/odom_broadcaster.py)
            # and you would add entry points here like:
            # 'odom_broadcaster_node = my_robot_bringup.odom_broadcaster:main',
        ],
    },
)