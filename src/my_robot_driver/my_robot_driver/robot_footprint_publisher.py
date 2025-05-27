#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PolygonStamped, Point32
from tf2_ros import Buffer, TransformListener
from rclpy.duration import Duration as rclpyDuration
from rclpy.time import Time as rclpyTime
import math

class RobotFootprintPublisher(Node):
    def __init__(self):
        super().__init__('robot_footprint_publisher')
        
        # Parameters
        self.declare_parameter('robot_length', 0.35)  # 35cm
        self.declare_parameter('robot_width', 0.20)   # 20cm 
        self.declare_parameter('robot_base_frame', 'base_link')
        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('publish_frequency', 5.0)  # Hz
        self.declare_parameter('footprint_topic', '/robot_footprint')
        
        self.robot_length = self.get_parameter('robot_length').value
        self.robot_width = self.get_parameter('robot_width').value
        self.robot_base_frame = self.get_parameter('robot_base_frame').value
        self.global_frame = self.get_parameter('global_frame').value
        self.publish_freq = self.get_parameter('publish_frequency').value
        self.footprint_topic = self.get_parameter('footprint_topic').value
        
        # TF setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Publisher
        self.footprint_pub = self.create_publisher(
            PolygonStamped, 
            self.footprint_topic, 
            10
        )
        
        # Timer
        self.timer = self.create_timer(
            1.0 / self.publish_freq, 
            self.publish_footprint
        )
        
        self.get_logger().info(
            f"Robot footprint publisher started. "
            f"Size: {self.robot_length:.2f}m x {self.robot_width:.2f}m. "
            f"Publishing to {self.footprint_topic} at {self.publish_freq:.1f} Hz"
        )

    def publish_footprint(self):
        try:
            # Get robot pose in global frame
            transform = self.tf_buffer.lookup_transform(
                self.global_frame, 
                self.robot_base_frame,
                rclpyTime(), 
                rclpyDuration(seconds=0.1)
            )
            
            robot_x = transform.transform.translation.x
            robot_y = transform.transform.translation.y
            
            # Get robot orientation (yaw)
            rot = transform.transform.rotation
            # Convert quaternion to yaw
            yaw = math.atan2(
                2.0 * (rot.w * rot.z + rot.x * rot.y),
                1.0 - 2.0 * (rot.y * rot.y + rot.z * rot.z)
            )
            
            # Create footprint polygon (rectangle around robot)
            half_length = self.robot_length / 2.0
            half_width = self.robot_width / 2.0
            
            # Define corners in robot frame (relative to base_link)
            corners_robot_frame = [
                (half_length, half_width),    # Front-left
                (half_length, -half_width),   # Front-right  
                (-half_length, -half_width),  # Back-right
                (-half_length, half_width)    # Back-left
            ]
            
            # Transform corners to global frame
            polygon = PolygonStamped()
            polygon.header.stamp = self.get_clock().now().to_msg()
            polygon.header.frame_id = self.global_frame
            
            cos_yaw = math.cos(yaw)
            sin_yaw = math.sin(yaw)
            
            for corner_x, corner_y in corners_robot_frame:
                # Rotate and translate corner to global frame
                global_x = robot_x + (corner_x * cos_yaw - corner_y * sin_yaw)
                global_y = robot_y + (corner_x * sin_yaw + corner_y * cos_yaw)
                
                point = Point32()
                point.x = float(global_x)
                point.y = float(global_y) 
                point.z = 0.0
                polygon.polygon.points.append(point)
            
            self.footprint_pub.publish(polygon)
            
        except Exception as e:
            self.get_logger().warn(
                f"Failed to publish robot footprint: {e}", 
                throttle_duration_sec=2.0
            )

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = RobotFootprintPublisher()
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node: 
            node.get_logger().info("Robot footprint publisher shutting down.")
    except Exception as e:
        if node: 
            node.get_logger().error(f"Unhandled exception: {e}")
        else: 
            print(f"Unhandled exception before node init: {e}")
    finally:
        if node and rclpy.ok():
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
