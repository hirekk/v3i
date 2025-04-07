from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

def plot_weight_evolution(experiment_path: Path) -> go.Figure:
    """Create an interactive plot of weight component evolution."""
    with open(experiment_path / "experiment.json", "r") as f:
        data = json.load(f)
    
    # Extract weight history
    steps = []
    w_comp = []
    x_comp = []
    y_comp = []
    z_comp = []
    
    for record in data["weight_history"]:
        steps.append(record["epoch"] + record["step"]/20000)  # normalize step to epoch
        w_comp.append(record["w"])
        x_comp.append(record["x"])
        y_comp.append(record["y"])
        z_comp.append(record["z"])
    
    # Create figure
    fig = go.Figure()
    
    # Add traces for each component
    fig.add_trace(go.Scatter(
        x=steps, y=w_comp,
        mode='lines',
        name='w (real)',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=steps, y=x_comp,
        mode='lines',
        name='x (i)',
        line=dict(color='red')
    ))
    fig.add_trace(go.Scatter(
        x=steps, y=y_comp,
        mode='lines',
        name='y (j)',
        line=dict(color='green')
    ))
    fig.add_trace(go.Scatter(
        x=steps, y=z_comp,
        mode='lines',
        name='z (k)',
        line=dict(color='purple')
    ))
    
    # Update layout
    fig.update_layout(
        title='Weight Components Evolution',
        xaxis_title='Epoch',
        yaxis_title='Component Value',
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig

def launch_dashboard() -> None:
    """Launch Streamlit dashboard for experiment visualization."""
    st.set_page_config(page_title="Quaternion Perceptron Dashboard", layout="wide")
    st.title("Quaternion Perceptron Weight Evolution")
    
    # Experiment selection
    experiments_dir = Path("data/experiments")
    experiments = sorted([d for d in experiments_dir.iterdir() if d.is_dir()], reverse=True)
    
    if not experiments:
        st.error("No experiments found in data/experiments directory")
        return
        
    selected_exp = st.selectbox(
        "Select experiment:",
        experiments,
        format_func=lambda x: x.name
    )
    
    # Plot weight evolution
    fig = plot_weight_evolution(selected_exp)
    st.plotly_chart(fig, use_container_width=True)
    
    # Display experiment details
    with open(selected_exp / "experiment.json", "r") as f:
        data = json.load(f)
    
    # Show final accuracies
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Final Training Accuracy", f"{data['train_accuracies'][-1]:.4f}")
    with col2:
        st.metric("Final Test Accuracy", f"{data['test_accuracies'][-1]:.4f}")

if __name__ == "__main__":
    launch_dashboard()
