import numpy as np
from typing import Tuple, List, Dict, Any
from abc import ABC, abstractmethod

class SystemModel(ABC):
    """Abstract base class for system models."""
    
    def __init__(self, dt: float = 0.01):
        self.dt = dt
        self.state = 0.0
        self.output_history = []
        self.time_history = []
        
    @abstractmethod
    def step(self, input_signal: float, time: float) -> float:
        """Simulate one step of the system."""
        pass
    
    @abstractmethod
    def reset(self):
        """Reset system state."""
        pass
    
    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Get system information."""
        pass

class FirstOrderSystem(SystemModel):
    """
    First-order system: G(s) = K / (τs + 1)
    
    Transfer function: Y(s)/U(s) = K / (τs + 1)
    State equation: τ * dy/dt + y = K * u
    """
    
    def __init__(self, K: float = 1.0, tau: float = 1.0, dt: float = 0.01):
        super().__init__(dt)
        self.K = K  # Gain
        self.tau = tau  # Time constant
        self.state = 0.0
        
    def step(self, input_signal: float, time: float) -> float:
        """
        Simulate one step using Euler integration.
        
        State equation: τ * dy/dt + y = K * u
        Rearranged: dy/dt = (K*u - y) / τ
        """
        # Euler integration
        derivative = (self.K * input_signal - self.state) / self.tau
        self.state += derivative * self.dt
        
        # Store history
        self.output_history.append(self.state)
        self.time_history.append(time)
        
        return self.state
    
    def reset(self):
        """Reset system state."""
        self.state = 0.0
        self.output_history = []
        self.time_history = []
    
    def get_info(self) -> Dict[str, Any]:
        return {
            'type': 'First Order',
            'transfer_function': f'G(s) = {self.K} / ({self.tau}s + 1)',
            'parameters': {
                'K': self.K,
                'tau': self.tau
            },
            'characteristics': {
                'time_constant': self.tau,
                'steady_state_gain': self.K,
                'settling_time': 4 * self.tau
            }
        }

class SecondOrderSystem(SystemModel):
    """
    Second-order system: G(s) = ωn² / (s² + 2ζωns + ωn²)
    
    Transfer function: Y(s)/U(s) = ωn² / (s² + 2ζωns + ωn²)
    State equations: 
        dx1/dt = x2
        dx2/dt = -ωn²*x1 - 2ζωn*x2 + ωn²*u
    """
    
    def __init__(self, omega_n: float = 1.0, zeta: float = 0.7, dt: float = 0.01):
        super().__init__(dt)
        self.omega_n = omega_n  # Natural frequency
        self.zeta = zeta  # Damping ratio
        self.state = np.array([0.0, 0.0])  # [position, velocity]
        
    def step(self, input_signal: float, time: float) -> float:
        """
        Simulate one step using Euler integration.
        
        State equations:
            dx1/dt = x2
            dx2/dt = -ωn²*x1 - 2ζωn*x2 + ωn²*u
        """
        # Current state
        x1, x2 = self.state
        
        # Derivatives
        dx1_dt = x2
        dx2_dt = (-self.omega_n**2 * x1 - 
                  2 * self.zeta * self.omega_n * x2 + 
                  self.omega_n**2 * input_signal)
        
        # Euler integration
        self.state[0] += dx1_dt * self.dt
        self.state[1] += dx2_dt * self.dt
        
        # Store history
        self.output_history.append(self.state[0])
        self.time_history.append(time)
        
        return self.state[0]
    
    def reset(self):
        """Reset system state."""
        self.state = np.array([0.0, 0.0])
        self.output_history = []
        self.time_history = []
    
    def get_info(self) -> Dict[str, Any]:
        return {
            'type': 'Second Order',
            'transfer_function': f'G(s) = {self.omega_n**2} / (s² + {2*self.zeta*self.omega_n}s + {self.omega_n**2})',
            'parameters': {
                'omega_n': self.omega_n,
                'zeta': self.zeta
            },
            'characteristics': {
                'natural_frequency': self.omega_n,
                'damping_ratio': self.zeta,
                'settling_time': 4 / (self.zeta * self.omega_n) if self.zeta > 0 else float('inf'),
                'peak_time': np.pi / (self.omega_n * np.sqrt(1 - self.zeta**2)) if self.zeta < 1 else float('inf')
            }
        }

class TimeDelaySystem(SystemModel):
    """
    First-order system with time delay: G(s) = K * e^(-Ts) / (τs + 1)
    
    Transfer function: Y(s)/U(s) = K * e^(-Ts) / (τs + 1)
    """
    
    def __init__(self, K: float = 1.0, tau: float = 1.0, delay: float = 0.5, dt: float = 0.01):
        super().__init__(dt)
        self.K = K  # Gain
        self.tau = tau  # Time constant
        self.delay = delay  # Time delay
        self.state = 0.0
        
        # Delay buffer
        self.delay_steps = int(self.delay / self.dt)
        self.input_buffer = [0.0] * self.delay_steps
        
    def step(self, input_signal: float, time: float) -> float:
        """
        Simulate one step with time delay.
        """
        # Add input to delay buffer
        delayed_input = self.input_buffer.pop(0)
        self.input_buffer.append(input_signal)
        
        # Simulate first-order system with delayed input
        derivative = (self.K * delayed_input - self.state) / self.tau
        self.state += derivative * self.dt
        
        # Store history
        self.output_history.append(self.state)
        self.time_history.append(time)
        
        return self.state
    
    def reset(self):
        """Reset system state."""
        self.state = 0.0
        self.input_buffer = [0.0] * self.delay_steps
        self.output_history = []
        self.time_history = []
    
    def get_info(self) -> Dict[str, Any]:
        return {
            'type': 'First Order with Time Delay',
            'transfer_function': f'G(s) = {self.K} * e^(-{self.delay}s) / ({self.tau}s + 1)',
            'parameters': {
                'K': self.K,
                'tau': self.tau,
                'delay': self.delay
            },
            'characteristics': {
                'time_constant': self.tau,
                'time_delay': self.delay,
                'steady_state_gain': self.K,
                'settling_time': 4 * self.tau + self.delay
            }
        }

class IntegratorSystem(SystemModel):
    """
    Pure integrator system: G(s) = K / s
    
    Transfer function: Y(s)/U(s) = K / s
    State equation: dy/dt = K * u
    """
    
    def __init__(self, K: float = 1.0, dt: float = 0.01):
        super().__init__(dt)
        self.K = K  # Gain
        self.state = 0.0
        
    def step(self, input_signal: float, time: float) -> float:
        """
        Simulate one step using Euler integration.
        """
        # Integrate input
        self.state += self.K * input_signal * self.dt
        
        # Store history
        self.output_history.append(self.state)
        self.time_history.append(time)
        
        return self.state
    
    def reset(self):
        """Reset system state."""
        self.state = 0.0
        self.output_history = []
        self.time_history = []
    
    def get_info(self) -> Dict[str, Any]:
        return {
            'type': 'Integrator',
            'transfer_function': f'G(s) = {self.K} / s',
            'parameters': {
                'K': self.K
            },
            'characteristics': {
                'type': 'Integrator',
                'steady_state_gain': float('inf')
            }
        }

def get_system_models() -> Dict[str, SystemModel]:
    """
    Get a dictionary of available system models.
    
    Returns:
        Dictionary mapping system names to system model instances
    """
    return {
        'First Order (τ=1)': FirstOrderSystem(K=1.0, tau=1.0),
        'First Order (τ=2)': FirstOrderSystem(K=1.0, tau=2.0),
        'Second Order (ζ=0.7)': SecondOrderSystem(omega_n=1.0, zeta=0.7),
        'Second Order (ζ=0.3)': SecondOrderSystem(omega_n=1.0, zeta=0.3),
        'Second Order (ζ=1.2)': SecondOrderSystem(omega_n=1.0, zeta=1.2),
        'Time Delay': TimeDelaySystem(K=1.0, tau=1.0, delay=0.5),
        'Integrator': IntegratorSystem(K=1.0)
    }

def simulate_system(system: SystemModel, controller, t_final: float = 10.0, 
                   dt: float = 0.01) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Simulate a system with PID control.
    
    Args:
        system: System model to simulate
        controller: PID controller
        t_final: Final simulation time
        dt: Time step
        
    Returns:
        Tuple of (time, setpoint, output, control_signal)
    """
    # Reset system and controller
    system.reset()
    controller.reset()
    
    # Simulation arrays
    time_points = np.arange(0, t_final + dt, dt)
    setpoint_array = []
    output_array = []
    control_array = []
    
    for t in time_points:
        # Get current measurement
        if hasattr(system, 'state'):
            if isinstance(system.state, np.ndarray):
                measurement = system.state[0]  # For second-order systems, use first state variable
            else:
                measurement = system.state
        else:
            measurement = system.output_history[-1] if system.output_history else 0.0
        
        # Compute control signal
        control = controller.compute(measurement, t)
        
        # Apply control to system
        output = system.step(control, t)
        
        # Store results
        setpoint_array.append(controller.setpoint)
        output_array.append(output)
        control_array.append(control)
    
    return time_points.tolist(), setpoint_array, output_array, control_array 