import numpy as np
from typing import Tuple, List

class PIDController:
    """
    PID Controller implementation with anti-windup and derivative filtering.
    
    Attributes:
        Kp (float): Proportional gain
        Ki (float): Integral gain  
        Kd (float): Derivative gain
        setpoint (float): Target value
        integral (float): Accumulated integral error
        prev_error (float): Previous error for derivative calculation
        dt (float): Time step
        output_limits (tuple): Min/max output limits
        integral_limits (tuple): Min/max integral limits
        derivative_filter (float): Low-pass filter coefficient for derivative
    """
    
    def __init__(self, Kp: float = 1.0, Ki: float = 0.0, Kd: float = 0.0, 
                 setpoint: float = 0.0, dt: float = 0.01,
                 output_limits: Tuple[float, float] = (-100, 100),
                 integral_limits: Tuple[float, float] = (-50, 50),
                 derivative_filter: float = 0.1):
        """
        Initialize PID Controller.
        
        Args:
            Kp: Proportional gain
            Ki: Integral gain
            Kd: Derivative gain
            setpoint: Target value
            dt: Time step
            output_limits: Min/max output limits
            integral_limits: Min/max integral limits
            derivative_filter: Low-pass filter coefficient (0-1)
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.dt = dt
        
        # State variables
        self.integral = 0.0
        self.prev_error = 0.0
        self.filtered_derivative = 0.0
        
        # Limits
        self.output_limits = output_limits
        self.integral_limits = integral_limits
        self.derivative_filter = derivative_filter
        
        # History for plotting
        self.error_history = []
        self.output_history = []
        self.integral_history = []
        self.derivative_history = []
        self.time_history = []
        
    def set_gains(self, Kp: float, Ki: float, Kd: float):
        """Update PID gains."""
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        
    def set_setpoint(self, setpoint: float):
        """Update setpoint."""
        self.setpoint = setpoint
        
    def reset(self):
        """Reset controller state."""
        self.integral = 0.0
        self.prev_error = 0.0
        self.filtered_derivative = 0.0
        self.error_history = []
        self.output_history = []
        self.integral_history = []
        self.derivative_history = []
        self.time_history = []
        
    def compute(self, measurement: float, time: float) -> float:
        """
        Compute control signal.
        
        Args:
            measurement: Current system output
            time: Current time
            
        Returns:
            Control signal
        """
        # Calculate error
        error = self.setpoint - measurement
        
        # Proportional term
        proportional = self.Kp * error
        
        # Integral term with anti-windup
        self.integral += error * self.dt
        self.integral = np.clip(self.integral, self.integral_limits[0], self.integral_limits[1])
        integral = self.Ki * self.integral
        
        # Derivative term with filtering
        derivative_raw = (error - self.prev_error) / self.dt
        self.filtered_derivative = (self.derivative_filter * derivative_raw + 
                                  (1 - self.derivative_filter) * self.filtered_derivative)
        derivative = self.Kd * self.filtered_derivative
        
        # Total control signal
        output = proportional + integral + derivative
        
        # Apply output limits
        output = np.clip(output, self.output_limits[0], self.output_limits[1])
        
        # Update previous error
        self.prev_error = error
        
        # Store history
        self.error_history.append(error)
        self.output_history.append(output)
        self.integral_history.append(integral)
        self.derivative_history.append(derivative)
        self.time_history.append(time)
        
        return output
    
    def get_performance_metrics(self) -> dict:
        """
        Calculate performance metrics from the response.
        
        Returns:
            Dictionary containing performance metrics
        """
        if len(self.error_history) < 10:
            return {
                'rise_time': None,
                'settling_time': None,
                'overshoot': None,
                'steady_state_error': None,
                'peak_time': None
            }
        
        # Convert to numpy arrays for calculations
        time = np.array(self.time_history)
        error = np.array(self.error_history)
        output = np.array(self.output_history)
        
        # Steady state error (last 10% of data)
        n_steady = max(1, len(error) // 10)
        steady_state_error = np.mean(error[-n_steady:])
        
        # Find rise time (time to reach 90% of setpoint)
        target_90 = 0.9 * self.setpoint
        rise_indices = np.where(np.abs(output) >= np.abs(target_90))[0]
        rise_time = time[rise_indices[0]] if len(rise_indices) > 0 else None
        
        # Find settling time (time to stay within 5% of setpoint)
        tolerance = 0.05 * abs(self.setpoint)
        settled_indices = np.where(np.abs(error) <= tolerance)[0]
        settling_time = time[settled_indices[0]] if len(settled_indices) > 0 else None
        
        # Find overshoot
        if self.setpoint > 0:
            max_output = np.max(output)
            overshoot = ((max_output - self.setpoint) / self.setpoint) * 100 if max_output > self.setpoint else 0
        else:
            min_output = np.min(output)
            overshoot = ((min_output - self.setpoint) / self.setpoint) * 100 if min_output < self.setpoint else 0
        
        # Find peak time
        if self.setpoint > 0:
            peak_idx = np.argmax(output)
        else:
            peak_idx = np.argmin(output)
        peak_time = time[peak_idx] if peak_idx < len(time) else None
        
        return {
            'rise_time': rise_time,
            'settling_time': settling_time,
            'overshoot': overshoot,
            'steady_state_error': steady_state_error,
            'peak_time': peak_time
        }
    
    def get_control_components(self) -> Tuple[List[float], List[float], List[float], List[float]]:
        """
        Get the individual control components for plotting.
        
        Returns:
            Tuple of (proportional, integral, derivative, total) components
        """
        if not self.error_history:
            return [], [], [], []
        
        proportional = [self.Kp * e for e in self.error_history]
        integral = self.integral_history
        derivative = self.derivative_history
        total = self.output_history
        
        return proportional, integral, derivative, total 