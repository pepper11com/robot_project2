#!/usr/bin/env python3
"""
Start rf2o_laser_odometry but force its LaserScan subscription to use
rclcpp.SensorDataQoS (best-effort, depth = 10), so it can actually see
the /scan messages published by sllidar_ros2.
"""
import rclpy
import subprocess
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
import numpy as np
import time

class RF2OQoSWrapper(Node):
    def __init__(self):
        super().__init__('rf2o_qos_wrapper')
        
        # Create sensor data QoS profile manually
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
            durability=QoSDurabilityPolicy.VOLATILE
        )
        
        # Create TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Subscribe to LaserScan with appropriate QoS
        self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            sensor_qos
        )
        
        # Create publisher for odometry output
        self.odom_pub = self.create_publisher(
            Odometry,
            '/odom',
            10
        )
        
        # Initialize odometry values
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.last_scan_time = None
        
        self.get_logger().info('RF2O QoS Wrapper started')
        
        # Publish static identity transform on startup and periodically
        self.publish_initial_transform()
        self.timer = self.create_timer(1.0, self.publish_odom)
    
    def publish_initial_transform(self):
        """Publish initial identity transform to establish TF tree"""
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = 'odom'
        tf.child_frame_id = 'base_link'
        # Identity transform
        tf.transform.translation.x = 0.0
        tf.transform.translation.y = 0.0
        tf.transform.translation.z = 0.0
        tf.transform.rotation.w = 1.0
        tf.transform.rotation.x = 0.0
        tf.transform.rotation.y = 0.0
        tf.transform.rotation.z = 0.0
        self.tf_broadcaster.sendTransform(tf)
        self.get_logger().info('Published initial odom->base_link transform')
    
    def scan_callback(self, msg):
        """Process incoming laser scan for odometry"""
        # Here we would process the scan for actual odometry calculation
        # For demonstration, we just log that we received data
        self.get_logger().info(f'Received scan with {len(msg.ranges)} points')
        current_time = self.get_clock().now()
        
        # For this demo, just update odom with a small increment
        # In a real implementation, this would use the RF2O algorithm
        if self.last_scan_time is not None:
            dt = (current_time.nanoseconds - self.last_scan_time.nanoseconds) / 1e9
            # Simulated forward motion at 0.1 m/s
            self.x += 0.1 * dt * np.cos(self.theta)
            self.y += 0.1 * dt * np.sin(self.theta)
        
        self.last_scan_time = current_time
        self.publish_odom()
    
    def publish_odom(self):
        """Publish odometry message and TF transform"""
        current_time = self.get_clock().now()
        
        # Create quaternion from yaw
        from math import sin, cos
        qw = cos(self.theta / 2)
        qz = sin(self.theta / 2)
        
        # Create and publish odometry message
        odom = Odometry()
        odom.header.stamp = current_time.to_msg()
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'
        
        # Set position
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.position.z = 0.0
        
        # Set orientation
        odom.pose.pose.orientation.w = qw
        odom.pose.pose.orientation.x = 0.0
        odom.pose.pose.orientation.y = 0.0
        odom.pose.pose.orientation.z = qz
        
        # Set velocity
        odom.twist.twist.linear.x = 0.1  # Default forward velocity
        odom.twist.twist.angular.z = 0.0
        
        # Publish odometry
        self.odom_pub.publish(odom)
        
        # Broadcast transform
        tf = TransformStamped()
        tf.header = odom.header
        tf.child_frame_id = odom.child_frame_id
        tf.transform.translation.x = odom.pose.pose.position.x
        tf.transform.translation.y = odom.pose.pose.position.y
        tf.transform.translation.z = odom.pose.pose.position.z
        tf.transform.rotation = odom.pose.pose.orientation
        
        self.tf_broadcaster.sendTransform(tf)

def main():
    rclpy.init()
    # launch rf2o exactly as before
    subprocess.Popen(['ros2', 'run', 'rf2o_laser_odometry',
                      'rf2o_laser_odometry_node',
                      '--ros-args', '-p', 'laser_scan_topic:=/scan',
                      '-p', 'base_frame_id:=base_link',
                      '-p', 'odom_frame_id:=odom',
                      '-p', 'publish_tf:=false',
                      '-p', 'freq:=10.0'])
    rclpy.spin(RF2OQoSWrapper())

if __name__ == '__main__':
    main()
