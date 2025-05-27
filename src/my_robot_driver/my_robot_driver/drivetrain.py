# my_robot_driver/drivetrain.py
"""
Drivetrain helper for H-bridges like L298N using gpiozero.Motor.
Assumes three pins per motor (forward, backward, enable for PWM).
"""
from gpiozero import Motor, GPIOPinMissing # For specific exception
from gpiozero.exc import BadPinFactory # For pigpio not running
from time import sleep

class Drivetrain:
    def __init__(self, logger=None):
        self.logger = logger
        self._log_info = logger.info if logger else print
        self._log_error = logger.error if logger else print
        self._log_fatal = logger.fatal if logger else print

        # ----- PIN MAP (BCM numbering) -----
        # These pins correspond to the "original" left and right motors
        # as they are physically wired to the H-bridge.
        LEFT_MOTOR_FORWARD_PIN = 27
        LEFT_MOTOR_BACKWARD_PIN = 17
        LEFT_MOTOR_ENABLE_PIN = 25  # PWM

        RIGHT_MOTOR_FORWARD_PIN = 22
        RIGHT_MOTOR_BACKWARD_PIN = 23
        RIGHT_MOTOR_ENABLE_PIN = 24 # PWM
        # --------------------------------------------------------------------

        try:
            self.left_motor = Motor(
                forward=LEFT_MOTOR_FORWARD_PIN,
                backward=LEFT_MOTOR_BACKWARD_PIN,
                enable=LEFT_MOTOR_ENABLE_PIN,
                pwm=True
            )
            self.right_motor = Motor(
                forward=RIGHT_MOTOR_FORWARD_PIN,
                backward=RIGHT_MOTOR_BACKWARD_PIN,
                enable=RIGHT_MOTOR_ENABLE_PIN,
                pwm=True
            )
            self._log_info("Drivetrain initialised (gpiozero.Motor with enable pins).")
        except BadPinFactory:
            msg = ("gpiozero BadPinFactory error. Is pigpiod service running? "
                   "Run 'sudo systemctl start pigpiod' or ensure it starts on boot.")
            self._log_fatal(msg)
            raise RuntimeError(msg) # Propagate to stop node init
        except GPIOPinMissing:
            msg = "gpiozero GPIOPinMissing error. One or more pins are not available. Check pin numbers."
            self._log_fatal(msg)
            raise RuntimeError(msg)
        except Exception as e:
            msg = f"Failed to initialize gpiozero.Motor. Error: {e}"
            self._log_fatal(msg)
            raise RuntimeError(msg)

    def set_speeds(self, left_pwm: float, right_pwm: float):
        """
        Set wheel speeds as PWM values.
        Args:
            left_pwm (float): -1.0 (full reverse) to +1.0 (full forward) for the motor
                              physically wired as "left_motor" above.
            right_pwm (float): -1.0 (full reverse) to +1.0 (full forward) for the motor
                               physically wired as "right_motor" above.
        """
        safe_left_pwm = max(-1.0, min(1.0, left_pwm))
        safe_right_pwm = max(-1.0, min(1.0, right_pwm))

        # self.left_motor.value is for the motor connected to LEFT_MOTOR_..._PINS
        # self.right_motor.value is for the motor connected to RIGHT_MOTOR_..._PINS
        self.left_motor.value = safe_left_pwm
        self.right_motor.value = safe_right_pwm
        
    def stop(self):
        self.left_motor.stop()
        self.right_motor.stop()

    def close(self):
        self._log_info("Closing drivetrain GPIO resources.")
        self.stop() # Ensure motors are stopped before closing
        if hasattr(self.left_motor, 'close') and callable(self.left_motor.close):
            self.left_motor.close()
        if hasattr(self.right_motor, 'close') and callable(self.right_motor.close):
            self.right_motor.close()
        # A small delay might not be necessary here unless you observe issues
        # sleep(0.1) 