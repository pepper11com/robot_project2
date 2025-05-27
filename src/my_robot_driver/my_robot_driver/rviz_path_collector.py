#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import sys
import termios 
import tty     
import math
import threading 
import select 

class RvizPathCollectorNode(Node):
    def __init__(self):
        super().__init__('rviz_path_collector')

        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('rviz_clicked_point_topic', '/rviz_clicked_point')
        self.declare_parameter('output_path_topic', '/goal_path')

        self.map_frame = self.get_parameter('map_frame').value
        self.rviz_clicked_point_topic = self.get_parameter('rviz_clicked_point_topic').value
        self.output_path_topic = self.get_parameter('output_path_topic').value
        
        self.path_publisher = self.create_publisher(Path, self.output_path_topic, 10)
        self.goal_subscriber = self.create_subscription(
            PoseStamped,
            self.rviz_clicked_point_topic,
            self.rviz_goal_callback,
            10
        )

        self.collected_waypoints: list[PoseStamped] = []
        self.get_logger().info(
            f"RViz Path Collector started.\n"
            f"Ensure RViz '2D Nav Goal' tool publishes to: {self.rviz_clicked_point_topic}\n"
            f"Click in RViz to add waypoints.\n"
            f"In THIS terminal, type a command and press Enter:\n"
            f"  's' - SEND the collected path to {self.output_path_topic}\n"
            f"  'c' - CLEAR the current path\n"
            f"  'p' - PRINT the current path\n"
            f"  'q' - QUIT."
        )
        
        self.is_shutdown_requested = False
        self.original_terminal_settings = termios.tcgetattr(sys.stdin.fileno())

        self.input_thread = threading.Thread(target=self._keyboard_input_loop)
        self.input_thread.daemon = True 
        self.input_thread.start()

    # --- ROS Action Scheduling ---
    def _schedule_action(self, action_callable):
        """Schedules a given callable to be run by the node's executor."""
        if self.is_shutdown_requested:
            return
        # Create a timer that we'll manually cancel after first execution
        timer = self.create_timer(0.001, lambda: self._execute_scheduled_action(action_callable, timer))

    def _execute_scheduled_action(self, action_callable, timer):
        # Cancel the timer to make it one-shot
        timer.cancel()
        
        if not self.is_shutdown_requested:
            try:
                action_callable()
            except Exception as e:
                self.get_logger().error(f"Error executing scheduled action {action_callable.__name__}: {e}")


    def rviz_goal_callback(self, msg: PoseStamped):
        if self.is_shutdown_requested: return
        if msg.header.frame_id != self.map_frame:
            self.get_logger().warn(
                f"Received goal from RViz in frame '{msg.header.frame_id}' but collector expects '{self.map_frame}'. "
                "Ensure RViz Fixed Frame is 'map' when publishing goals."
            )
        if msg.pose.orientation.w == 0.0 and \
           msg.pose.orientation.x == 0.0 and \
           msg.pose.orientation.y == 0.0 and \
           msg.pose.orientation.z == 0.0:
            msg.pose.orientation.w = 1.0 
        self.collected_waypoints.append(msg)
        self.get_logger().info(
            f"Waypoint {len(self.collected_waypoints)} added: "
            f"X={msg.pose.position.x:.2f}, Y={msg.pose.position.y:.2f}, "
            f"Yaw={self.get_yaw_from_pose(msg):.2f} rad (in frame '{msg.header.frame_id}')"
        )
        self.get_logger().info(f"Current path has {len(self.collected_waypoints)} waypoints.")

    def get_yaw_from_pose(self, pose_stamped_msg: PoseStamped) -> float:
        q = pose_stamped_msg.pose.orientation
        try:
            from tf_transformations import euler_from_quaternion 
            _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
            return yaw
        except ImportError:
            return 2.0 * math.atan2(q.z, q.w)

    def publish_collected_path(self):
        if self.is_shutdown_requested: return
        if not self.collected_waypoints:
            self.get_logger().info("No waypoints collected to send.")
            return
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = self.map_frame 
        for original_pose_stamped in self.collected_waypoints:
            path_pose = PoseStamped()
            path_pose.header.stamp = path_msg.header.stamp 
            path_pose.header.frame_id = self.map_frame    
            path_pose.pose = original_pose_stamped.pose   
            path_msg.poses.append(path_pose)
        self.path_publisher.publish(path_msg)
        self.get_logger().info(f"Published path with {len(path_msg.poses)} waypoints to {self.output_path_topic}.")

    def clear_waypoints(self):
        if self.is_shutdown_requested: return
        self.collected_waypoints.clear()
        self.get_logger().info("Collected waypoints cleared.")

    def print_waypoints(self):
        if self.is_shutdown_requested: return
        if not self.collected_waypoints:
            self.get_logger().info("No waypoints to print.")
            return
        self.get_logger().info("Current collected waypoints:")
        for i, wp in enumerate(self.collected_waypoints):
            self.get_logger().info(
                f"  {i+1}: X={wp.pose.position.x:.2f}, Y={wp.pose.position.y:.2f}, Yaw={self.get_yaw_from_pose(wp):.2f}"
            )

    def _keyboard_input_loop(self):
        self.get_logger().info("Keyboard input thread started. Type command and press Enter.")
        
        while rclpy.ok() and not self.is_shutdown_requested:
            try:
                rlist, _, _ = select.select([sys.stdin], [], [], 0.1) 
                if rlist:
                    line = sys.stdin.readline().strip().lower()
                    if not line and not self.is_shutdown_requested: 
                        continue
                    
                    if not self.is_shutdown_requested:
                        if line == 's':
                            self.get_logger().info("CLI: 's' received. Scheduling path publish.")
                            self._schedule_action(self.publish_collected_path)
                        elif line == 'c':
                            self.get_logger().info("CLI: 'c' received. Scheduling waypoint clear.")
                            self._schedule_action(self.clear_waypoints)
                        elif line == 'p':
                            self.get_logger().info("CLI: 'p' received. Scheduling waypoint print.")
                            self._schedule_action(self.print_waypoints)
                        elif line == 'q':
                            self.get_logger().info("CLI: 'q' received. Initiating application shutdown.")
                            self.is_shutdown_requested = True
                            # Create shutdown timer without oneshot parameter
                            shutdown_timer = self.create_timer(0.001, lambda: self._shutdown_callback(shutdown_timer))
                            break 
            except EOFError: 
                self.get_logger().info("EOF on stdin, keyboard input loop exiting.")
                break
            except Exception as e:
                if not self.is_shutdown_requested: 
                    self.get_logger().error(f"Exception in keyboard input loop: {e}")
                break 
        
        self.get_logger().info("Keyboard input thread finished.")

    def _shutdown_callback(self, timer):
        """Helper to cancel the timer and call shutdown."""
        timer.cancel()
        self._initiate_rclpy_shutdown()
        
    def _initiate_rclpy_shutdown(self):
        """Helper to call rclpy.shutdown() from the main executor thread."""
        if rclpy.ok():
            self.get_logger().info("rclpy.shutdown() called from main thread via timer.")
            # This will cause rclpy.spin() to exit.
            # The main function's finally block will then handle node destruction.
            rclpy.shutdown(context=self.context)


    def destroy_node(self):
        self.get_logger().info("RvizPathCollectorNode preparing to shut down.")
        self.is_shutdown_requested = True 
        
        if hasattr(self, 'input_thread') and self.input_thread.is_alive():
            self.get_logger().info("Attempting to join keyboard input thread...")
            self.input_thread.join(timeout=0.5) 
            if self.input_thread.is_alive():
                self.get_logger().warn("Keyboard input thread did not join cleanly.")
        
        try:
            if hasattr(self, 'original_terminal_settings'): # Check if it was set
                 termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self.original_terminal_settings)
                 self.get_logger().info("Original terminal settings restored.")
        except Exception as e:
            self.get_logger().error(f"Failed to restore terminal settings: {e}")

        super().destroy_node()
        self.get_logger().info("RvizPathCollectorNode destroyed.")


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = RvizPathCollectorNode()
        rclpy.spin(node)
    except KeyboardInterrupt: # This will catch Ctrl+C
        if node: node.get_logger().info("KeyboardInterrupt received by main, initiating shutdown.")
        else: print("KeyboardInterrupt received by main before node init.")
    # rclpy.spin() will exit if rclpy.shutdown() is called (e.g., by our 'q' command)
    # or if the node is removed from the executor and no other nodes are spinning on it.
    except Exception as e:
        if node: node.get_logger().error(f"Unhandled exception during spin: {e}")
        else: print(f"Unhandled exception before node init: {e}")
    finally:
        if node is not None and rclpy.ok() and node.context.is_valid(): # Ensure node exists and context is valid
            # If shutdown was initiated by 'q', rclpy might already be shutting down
            # node.destroy_node() will try to restore terminal settings
            if not node.is_shutdown_requested : # If 'q' wasn't pressed, signal shutdown
                node.is_shutdown_requested = True
            node.destroy_node() 
        
        if rclpy.ok(): # Check if rclpy hasn't been shut down yet
            rclpy.shutdown()
        print("RvizPathCollectorNode main process terminated.")

if __name__ == '__main__':
    main()