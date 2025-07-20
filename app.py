import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

# Import our custom modules
from pid_controller import PIDController
from system_models import get_system_models, simulate_system

# Initialize the Dash app
app = dash.Dash(__name__, 
                external_stylesheets=['https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css'])

# Get available system models
system_models = get_system_models()
system_names = list(system_models.keys())

# Initialize default values
default_Kp = 1.0
default_Ki = 0.0
default_Kd = 0.0
default_setpoint = 1.0
default_simulation_time = 10.0

# Create initial PID controller and system
pid_controller = PIDController(Kp=default_Kp, Ki=default_Ki, Kd=default_Kd, setpoint=default_setpoint)
current_system = system_models[system_names[0]]

# App layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("🎛️ Interactive PID Control Visualizer", className="header"),
        html.P("Explore the effects of Proportional, Integral, and Derivative gains on system response", className="header")
    ], className="header"),
    
    # Main container
    html.Div([
        # Control Panel
        html.Div([
            html.H3("🎮 Control Panel", className="control-panel"),
            
            # PID Gains Sliders
            html.Div([
                # Proportional Gain
                html.Div([
                    html.Div([
                        html.I(className="fas fa-chart-line", style={'color': '#3182ce'}),
                        html.Span("Proportional Gain (Kp)", className="slider-label")
                    ]),
                    dcc.Slider(
                        id='kp-slider',
                        min=0.0,
                        max=10.0,
                        step=0.1,
                        value=default_Kp,
                        marks={i: str(i) for i in range(0, 11, 2)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    html.Div(id='kp-value', className="slider-value")
                ], className="slider-group proportional"),
                
                # Integral Gain
                html.Div([
                    html.Div([
                        html.I(className="fas fa-chart-area", style={'color': '#38a169'}),
                        html.Span("Integral Gain (Ki)", className="slider-label")
                    ]),
                    dcc.Slider(
                        id='ki-slider',
                        min=0.0,
                        max=5.0,
                        step=0.1,
                        value=default_Ki,
                        marks={i: str(i) for i in range(0, 6, 1)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    html.Div(id='ki-value', className="slider-value")
                ], className="slider-group integral"),
                
                # Derivative Gain
                html.Div([
                    html.Div([
                        html.I(className="fas fa-chart-bar", style={'color': '#e53e3e'}),
                        html.Span("Derivative Gain (Kd)", className="slider-label")
                    ]),
                    dcc.Slider(
                        id='kd-slider',
                        min=0.0,
                        max=2.0,
                        step=0.05,
                        value=default_Kd,
                        marks={i: str(i) for i in range(0, 3, 1)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    html.Div(id='kd-value', className="slider-value")
                ], className="slider-group derivative")
            ], className="slider-container"),
            
            # System Selection and Parameters
            html.Div([
                html.Div([
                    html.H4("🔧 System Configuration", className="system-selection"),
                    
                    # System Selection
                    html.Div([
                        html.Label("Select System Model:", className="control-group"),
                        dcc.Dropdown(
                            id='system-dropdown',
                            options=[{'label': name, 'value': name} for name in system_names],
                            value=system_names[0],
                            clearable=False,
                            style={'marginBottom': '15px'}
                        )
                    ], className="control-group"),
                    
                    # Setpoint and Simulation Parameters
                    html.Div([
                        html.Div([
                            html.Label("Setpoint:", className="control-group"),
                            dcc.Input(
                                id='setpoint-input',
                                type='number',
                                value=default_setpoint,
                                step=0.1,
                                className="dash-input"
                            )
                        ], className="control-group"),
                        
                        html.Div([
                            html.Label("Simulation Time (s):", className="control-group"),
                            dcc.Input(
                                id='simulation-time-input',
                                type='number',
                                value=default_simulation_time,
                                step=0.5,
                                min=1.0,
                                max=30.0,
                                className="dash-input"
                            )
                        ], className="control-group")
                    ], className="simulation-controls")
                ], className="system-selection"),
                
                # Control Buttons
                html.Div([
                    html.Button("🔄 Reset", id='reset-button', className="btn btn-secondary"),
                    html.Button("▶️ Run Simulation", id='run-button', className="btn btn-primary"),
                    html.Button("📊 Export Data", id='export-button', className="btn btn-secondary")
                ], className="button-container")
            ])
        ], className="control-panel"),
        
        # Performance Metrics
        html.Div([
            html.H3("📈 Performance Metrics", className="metrics-container"),
            html.Div([
                html.Div([
                    html.Div(id='rise-time-value', className="metric-value"),
                    html.Div("Rise Time (s)", className="metric-label")
                ], className="metric-card"),
                html.Div([
                    html.Div(id='settling-time-value', className="metric-value"),
                    html.Div("Settling Time (s)", className="metric-label")
                ], className="metric-card"),
                html.Div([
                    html.Div(id='overshoot-value', className="metric-value"),
                    html.Div("Overshoot (%)", className="metric-label")
                ], className="metric-card"),
                html.Div([
                    html.Div(id='steady-error-value', className="metric-value"),
                    html.Div("Steady State Error", className="metric-label")
                ], className="metric-card")
            ], className="metrics-grid")
        ], className="metrics-container"),
        
        # Plots
        html.Div([
            html.H3("📊 System Response", className="plot-container"),
            
            # Main Response Plot
            dcc.Graph(
                id='response-plot',
                style={'height': '500px'},
                config={'displayModeBar': True, 'displaylogo': False}
            ),
            
            # Control Components Plot
            html.H4("🎛️ Control Signal Components", style={'marginTop': '30px', 'marginBottom': '20px'}),
            dcc.Graph(
                id='control-components-plot',
                style={'height': '400px'},
                config={'displayModeBar': True, 'displaylogo': False}
            )
        ], className="plot-container"),
        
        # Hidden divs for storing data
        html.Div(id='simulation-data', style={'display': 'none'}),
        html.Div(id='system-info', style={'display': 'none'})
        
    ], className="main-container")
])

# Callback to update slider value displays
@app.callback(
    [Output('kp-value', 'children'),
     Output('ki-value', 'children'),
     Output('kd-value', 'children')],
    [Input('kp-slider', 'value'),
     Input('ki-slider', 'value'),
     Input('kd-slider', 'value')]
)
def update_slider_values(kp, ki, kd):
    return f"Kp = {kp:.2f}", f"Ki = {ki:.2f}", f"Kd = {kd:.2f}"

# Callback to update system info
@app.callback(
    Output('system-info', 'children'),
    [Input('system-dropdown', 'value')]
)
def update_system_info(system_name):
    if system_name:
        system = system_models[system_name]
        info = system.get_info()
        return str(info)
    return ""

# Main simulation callback
@app.callback(
    [Output('response-plot', 'figure'),
     Output('control-components-plot', 'figure'),
     Output('simulation-data', 'children')],
    [Input('run-button', 'n_clicks'),
     Input('reset-button', 'n_clicks')],
    [State('kp-slider', 'value'),
     State('ki-slider', 'value'),
     State('kd-slider', 'value'),
     State('setpoint-input', 'value'),
     State('simulation-time-input', 'value'),
     State('system-dropdown', 'value')]
)
def update_simulation(run_clicks, reset_clicks, kp, ki, kd, setpoint, sim_time, system_name):
    ctx = callback_context
    if not ctx.triggered:
        # Initial load
        return create_empty_plots()
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'reset-button':
        return create_empty_plots()
    
    # Update controller and system
    global pid_controller, current_system
    
    if system_name:
        current_system = system_models[system_name]
    
    pid_controller.set_gains(kp, ki, kd)
    pid_controller.set_setpoint(setpoint)
    
    # Run simulation
    time_points, setpoint_array, output_array, control_array = simulate_system(
        current_system, pid_controller, sim_time
    )
    
    # Create plots
    response_fig = create_response_plot(time_points, setpoint_array, output_array, control_array)
    components_fig = create_control_components_plot(pid_controller)
    
    # Store simulation data
    simulation_data = {
        'time': time_points,
        'setpoint': setpoint_array,
        'output': output_array,
        'control': control_array,
        'system_name': system_name,
        'gains': {'Kp': kp, 'Ki': ki, 'Kd': kd}
    }
    
    return response_fig, components_fig, str(simulation_data)

# Callback to update performance metrics
@app.callback(
    [Output('rise-time-value', 'children'),
     Output('settling-time-value', 'children'),
     Output('overshoot-value', 'children'),
     Output('steady-error-value', 'children')],
    [Input('response-plot', 'figure')]
)
def update_performance_metrics(figure):
    if not figure or not figure.get('data'):
        return "N/A", "N/A", "N/A", "N/A"
    
    # Get metrics from PID controller
    metrics = pid_controller.get_performance_metrics()
    
    rise_time = f"{metrics['rise_time']:.3f}s" if metrics['rise_time'] is not None else "N/A"
    settling_time = f"{metrics['settling_time']:.3f}s" if metrics['settling_time'] is not None else "N/A"
    overshoot = f"{metrics['overshoot']:.1f}%" if metrics['overshoot'] is not None else "N/A"
    steady_error = f"{metrics['steady_state_error']:.4f}" if metrics['steady_state_error'] is not None else "N/A"
    
    return rise_time, settling_time, overshoot, steady_error

def create_response_plot(time_points, setpoint_array, output_array, control_array):
    """Create the main response plot."""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('System Response', 'Control Signal'),
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )
    
    # System response
    fig.add_trace(
        go.Scatter(
            x=time_points,
            y=setpoint_array,
            mode='lines',
            name='Setpoint',
            line=dict(color='#2E86AB', width=2, dash='dash'),
            hovertemplate='Time: %{x:.2f}s<br>Setpoint: %{y:.3f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=time_points,
            y=output_array,
            mode='lines',
            name='System Output',
            line=dict(color='#A23B72', width=3),
            hovertemplate='Time: %{x:.2f}s<br>Output: %{y:.3f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Control signal
    fig.add_trace(
        go.Scatter(
            x=time_points,
            y=control_array,
            mode='lines',
            name='Control Signal',
            line=dict(color='#F18F01', width=2),
            hovertemplate='Time: %{x:.2f}s<br>Control: %{y:.3f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title='',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=600,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    # Update axes
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig.update_yaxes(title_text="Control Signal", row=2, col=1)
    
    return fig

def create_control_components_plot(controller):
    """Create the control components plot."""
    proportional, integral, derivative, total = controller.get_control_components()
    
    if not proportional:
        return create_empty_plot("Control Components")
    
    time_points = controller.time_history
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=time_points,
        y=proportional,
        mode='lines',
        name='Proportional (P)',
        line=dict(color='#3182ce', width=2),
        hovertemplate='Time: %{x:.2f}s<br>P: %{y:.3f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=time_points,
        y=integral,
        mode='lines',
        name='Integral (I)',
        line=dict(color='#38a169', width=2),
        hovertemplate='Time: %{x:.2f}s<br>I: %{y:.3f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=time_points,
        y=derivative,
        mode='lines',
        name='Derivative (D)',
        line=dict(color='#e53e3e', width=2),
        hovertemplate='Time: %{x:.2f}s<br>D: %{y:.3f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=time_points,
        y=total,
        mode='lines',
        name='Total Control',
        line=dict(color='#2d3748', width=3),
        hovertemplate='Time: %{x:.2f}s<br>Total: %{y:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='',
        xaxis_title="Time (s)",
        yaxis_title="Control Components",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=400,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

def create_empty_plots():
    """Create empty plots for initial state."""
    empty_response = go.Figure()
    empty_response.add_annotation(
        text="Click 'Run Simulation' to start",
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, color="gray")
    )
    empty_response.update_layout(
        title='System Response',
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        height=600
    )
    
    empty_components = go.Figure()
    empty_components.add_annotation(
        text="Control components will appear here",
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, color="gray")
    )
    empty_components.update_layout(
        title='Control Components',
        xaxis_title="Time (s)",
        yaxis_title="Control Signal",
        height=400
    )
    
    return empty_response, empty_components, ""

def create_empty_plot(title):
    """Create a single empty plot."""
    fig = go.Figure()
    fig.add_annotation(
        text="No data available",
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, color="gray")
    )
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Amplitude"
    )
    return fig

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050) 