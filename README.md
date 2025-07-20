# 🎛️ Interactive PID Control Visualizer

A modern, interactive web application for visualizing and understanding PID (Proportional-Integral-Derivative) control systems. Built with Plotly Dash, this tool allows you to dynamically adjust control parameters and immediately see their effects on system response.

## 🌟 Features

### Interactive Controls
- **Real-time Parameter Adjustment**: Sliders for P, I, and D gains with immediate visual feedback
- **Multiple System Models**: Choose from first-order, second-order, time-delay, and integrator systems
- **Flexible Setpoint Control**: Adjust target values and simulation parameters
- **System Comparison**: Test different system characteristics

### Advanced Visualization
- **Multi-panel Plots**: System response, control signal, and individual control components
- **Performance Metrics**: Real-time calculation of rise time, settling time, overshoot, and steady-state error
- **Interactive Charts**: Zoom, pan, hover, and export capabilities
- **Professional UI**: Modern, responsive design with intuitive controls

### Educational Features
- **Control Theory Learning**: Visual understanding of PID control effects
- **Parameter Tuning**: Learn how each gain affects system behavior
- **System Analysis**: Understand different system dynamics
- **Performance Evaluation**: Quantitative analysis of control performance

## 🚀 Quick Start

### Prerequisites
- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. **Clone or download the project**
   ```bash
   git clone <repository-url>
   cd pid_visualizer
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Open your browser**
   Navigate to `http://localhost:8050`

## 📖 Usage Guide

### Getting Started
1. **Select a System**: Choose from the dropdown menu of available system models
2. **Set Parameters**: Adjust the setpoint and simulation time as needed
3. **Tune PID Gains**: Use the sliders to adjust P, I, and D gains
4. **Run Simulation**: Click "Run Simulation" to see the results
5. **Analyze Results**: View the plots and performance metrics

### Understanding the Controls

#### Proportional Gain (Kp)
- **Effect**: Responds to current error
- **High Kp**: Faster response, but may cause overshoot
- **Low Kp**: Slower response, but more stable
- **Typical Range**: 0.1 - 10.0

#### Integral Gain (Ki)
- **Effect**: Eliminates steady-state error
- **High Ki**: Faster elimination of steady-state error, but may cause oscillations
- **Low Ki**: Slower elimination of steady-state error
- **Typical Range**: 0.0 - 5.0

#### Derivative Gain (Kd)
- **Effect**: Responds to rate of change of error
- **High Kd**: Reduces overshoot, but may amplify noise
- **Low Kd**: Less noise sensitivity, but more overshoot
- **Typical Range**: 0.0 - 2.0

### System Models

#### First-Order System
- **Transfer Function**: G(s) = K / (τs + 1)
- **Characteristics**: Simple, stable, no overshoot
- **Applications**: Temperature control, level control

#### Second-Order System
- **Transfer Function**: G(s) = ωn² / (s² + 2ζωns + ωn²)
- **Characteristics**: Can exhibit overshoot, oscillations
- **Applications**: Position control, motor control

#### Time-Delay System
- **Transfer Function**: G(s) = K * e^(-Ts) / (τs + 1)
- **Characteristics**: Time delay makes control more challenging
- **Applications**: Chemical processes, transportation systems

#### Integrator System
- **Transfer Function**: G(s) = K / s
- **Characteristics**: Accumulates input over time
- **Applications**: Position control, velocity control

## 📊 Performance Metrics

The application calculates and displays key performance indicators:

- **Rise Time**: Time to reach 90% of the setpoint
- **Settling Time**: Time to stay within 5% of the setpoint
- **Overshoot**: Maximum deviation above the setpoint (percentage)
- **Steady-State Error**: Final error after system settles

## 🎯 Tuning Guidelines

### Ziegler-Nichols Method (First Tuning)
1. Set Ki = 0, Kd = 0
2. Increase Kp until sustained oscillations occur
3. Note the critical gain (Kc) and period (Tc)
4. Use: Kp = 0.6*Kc, Ki = 1.2*Kc/Tc, Kd = 0.075*Kc*Tc

### Conservative Tuning
1. Start with low gains (Kp = 0.5, Ki = 0.1, Kd = 0.05)
2. Increase Kp until desired response speed
3. Add Ki to eliminate steady-state error
4. Add Kd to reduce overshoot

### Aggressive Tuning
1. Start with higher gains (Kp = 2.0, Ki = 0.5, Kd = 0.1)
2. Fine-tune based on performance metrics
3. Balance between speed and stability

## 🔧 Technical Details

### Architecture
- **Frontend**: Plotly Dash (Python web framework)
- **Visualization**: Plotly for interactive charts
- **Backend**: Python with NumPy for calculations
- **Styling**: Custom CSS with modern design

### Key Components
- `app.py`: Main Dash application and UI
- `pid_controller.py`: PID control algorithm implementation
- `system_models.py`: System simulation models
- `assets/style.css`: Custom styling

### Mathematical Foundation
The PID controller implements the standard control law:
```
u(t) = Kp * e(t) + Ki * ∫e(τ)dτ + Kd * (de/dt)
```

Where:
- u(t) is the control signal
- e(t) is the error (setpoint - measurement)
- Kp, Ki, Kd are the proportional, integral, and derivative gains

## 🎓 Educational Applications

This tool is perfect for:
- **Control Theory Courses**: Visual demonstration of PID concepts
- **Engineering Education**: Hands-on learning of control systems
- **Research**: Quick prototyping and analysis of control strategies
- **Professional Development**: Understanding control system behavior

## 🚀 Future Enhancements

Planned features include:
- 3D visualization of parameter space
- Machine learning-based tuning suggestions
- Frequency domain analysis
- Multi-input multi-output (MIMO) systems
- Hardware-in-the-loop simulation
- Export functionality (PDF, CSV, etc.)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Plotly Dash team for the excellent web framework
- Control theory community for educational resources
- Open source contributors who made this project possible

---

**Happy Tuning! 🎛️✨** 