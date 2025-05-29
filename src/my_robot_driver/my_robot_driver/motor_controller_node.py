#!/usr/bin/env python3
"""
Translate /cmd_vel geometry_msgs/Twist into PWM commands for a 4-wheel tank drive.
Turns only when linear velocity is (almost) zero.
Handles robot being physically oriented 180 degrees from its original design
(i.e., old back is new front, old left is new right, old right is new left).
"""

from __future__ import annotations
import math, rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from geometry_msgs.msg import Twist

from .drivetrain import (
    Drivetrain,
)  # Assuming drivetrain.py is in the same package directory


# ---------- helper ----------------------------------------------------------
def clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# ---------------------------------------------------------------------------


class MotorControllerNode(Node):
    def __init__(self):
        super().__init__("motor_controller_node")
        self.get_logger().info(
            "Motor controller starting (4WD – stationary turning – ROBOT FLIPPED 180 DEG)…"
        )

        # ---------- tunable parameters ------------------------------------
        self.declare_parameter("wheel_base", 0.13)
        self.declare_parameter("wheel_radius", 0.0325)
        self.declare_parameter("max_wheel_lin_speed", 0.026)
        self.declare_parameter("fixed_turn_pwm", 0.38)  # Increased from 0.48 to 0.65
        self.declare_parameter("fixed_forward_pwm", 0.38)  # Increased from 0.32 to 0.38
        self.declare_parameter("stationary_threshold", 0.01)
        self.declare_parameter("cmd_vel_timeout", 0.3)
        # ------------------------------------------------------------------

        self.wheel_base = self.get_parameter("wheel_base").value
        self.wheel_radius = self.get_parameter("wheel_radius").value
        self.max_wheel_lin_speed = max(
            0.001, self.get_parameter("max_wheel_lin_speed").value
        )
        self.fixed_turn_pwm = clip(self.get_parameter("fixed_turn_pwm").value, 0.0, 1.0)
        self.fixed_forward_pwm = clip(
            self.get_parameter("fixed_forward_pwm").value, 0.0, 1.0
        )
        self.stationary_threshold = self.get_parameter("stationary_threshold").value
        self.cmd_timeout = Duration(seconds=self.get_parameter("cmd_vel_timeout").value)

        try:
            self.drivetrain = Drivetrain(logger=self.get_logger())
        except Exception as e:
            self.get_logger().fatal(f"Drivetrain init failed: {e}")
            raise SystemExit

        self.cmd_sub = self.create_subscription(Twist, "/cmd_vel", self._on_cmd, 10)
        self.last_cmd = self.get_clock().now()
        self.watchdog = self.create_timer(0.1, self._wdog)
        self.get_logger().info("Motor controller successfully initialized.")

    def _on_cmd(self, msg: Twist):
        self.last_cmd = self.get_clock().now()

        # --- LOGIC FOR FLIPPED ROBOT ---
        # 1. Incoming linear.x for NEW FORWARD. Motors need OLD BACKWARD command.
        v_input = msg.linear.x
        v_for_motor_calc = -v_input

        # 2. Incoming angular.z for NEW LEFT TURN (CCW).
        w_cmd = msg.angular.z
        # --- END LOGIC FOR FLIPPED ROBOT ---

        turn_only = abs(v_input) < self.stationary_threshold

        # Use cmd_vel values as speed multipliers, but ensure minimum power
        pwm_l_orig = 0.0
        pwm_r_orig = 0.0

        if turn_only and abs(w_cmd) > 0.01:
            # Scale turn power based on angular velocity, but ensure much higher minimum
            turn_scale = min(
                abs(w_cmd) / 0.15, 1.0
            )  # Normalize to max expected angular vel
            actual_turn_pwm = max(
                self.fixed_turn_pwm * turn_scale, self.fixed_turn_pwm * 0.95
            )  # Min 90% of fixed (was 85%)

            if w_cmd > 0:  # Turn left (CCW)
                pwm_l_orig = -actual_turn_pwm  # Left motor backward
                pwm_r_orig = actual_turn_pwm  # Right motor forward
            else:  # Turn right (CW)
                pwm_l_orig = actual_turn_pwm  # Left motor forward
                pwm_r_orig = -actual_turn_pwm  # Right motor backward
        elif not turn_only and abs(v_input) > 0.001:
            # Scale forward power based on linear velocity, but ensure minimum
            forward_scale = min(
                abs(v_input) / 0.012, 1.0
            )  # Normalize to max expected linear vel
            actual_forward_pwm = max(
                self.fixed_forward_pwm * forward_scale, self.fixed_forward_pwm * 0.8
            )  # Min 80% of fixed

            if v_for_motor_calc > 0:  # Forward
                pwm_l_orig = actual_forward_pwm
                pwm_r_orig = actual_forward_pwm
            else:  # Backward
                pwm_l_orig = -actual_forward_pwm
                pwm_r_orig = -actual_forward_pwm
        # else: both motors stay at 0.0 (stopped)

        # --- Assign PWMs to Drivetrain based on NEW orientation ---
        pwm_for_drivetrain_left_arg = (
            pwm_l_orig  # This goes to drivetrain's "left_motor" (NEW RIGHT)
        )
        pwm_for_drivetrain_right_arg = (
            pwm_r_orig  # This goes to drivetrain's "right_motor" (NEW LEFT)
        )

        self.get_logger().info(
            f"CMD:lx={v_input:.3f},az={w_cmd:.3f} | turn_only={turn_only} | "
            f"PWM_SCALED:L={pwm_l_orig:.3f},R={pwm_r_orig:.3f}"
        )

        self.drivetrain.set_speeds(
            pwm_for_drivetrain_left_arg, pwm_for_drivetrain_right_arg
        )

    def _wdog(self):
        if (self.get_clock().now() - self.last_cmd) > self.cmd_timeout:
            self.drivetrain.stop()
            self.last_cmd = self.get_clock().now()

    def destroy_node(self):
        self.get_logger().info("Motor-controller shutdown – motors off")
        try:
            if hasattr(self, "drivetrain") and self.drivetrain:
                self.drivetrain.stop()
                self.drivetrain.close()
        except Exception as e:
            self.get_logger().error(f"Drivetrain cleanup error: {e}")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = MotorControllerNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node:
            node.get_logger().info(
                "Keyboard interrupt, shutting down motor controller."
            )
    except SystemExit:
        # Logger might not be available if __init__ failed early
        print("Motor controller CRITICAL error during init, shutting down.")
    finally:
        if node and isinstance(node, Node) and rclpy.ok():
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
