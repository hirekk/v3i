import json
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


def plot_weight_evolution(experiment_path: Path) -> go.Figure:
    """Create an interactive plot of weight component evolution."""
    with open(experiment_path / "experiment.json", encoding="utf-8") as f:
        data = json.load(f)

    # Extract weight history
    steps = []
    w_comp = []
    x_comp = []
    y_comp = []
    z_comp = []

    for record in data["weight_history"]:
        steps.append(record["epoch"] + record["step"] / 20000)  # normalize step to epoch
        w_comp.append(record["w"])
        x_comp.append(record["x"])
        y_comp.append(record["y"])
        z_comp.append(record["z"])

    # Create figure
    fig = go.Figure()

    # Add traces for each component
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=w_comp,
            mode="lines",
            name="w (real)",
            line={"color": "blue"},
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=x_comp,
            mode="lines",
            name="x (i)",
            line={"color": "red"},
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=y_comp,
            mode="lines",
            name="y (j)",
            line={"color": "green"},
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=z_comp,
            mode="lines",
            name="z (k)",
            line={"color": "purple"},
        ),
    )

    # Update layout
    fig.update_layout(
        title="Weight Components Evolution",
        xaxis_title="Epoch",
        yaxis_title="Component Value",
        hovermode="x unified",
        template="plotly_white",
    )

    return fig


def plot_accuracy_comparison(experiment_path: Path) -> go.Figure:
    """Create comparison plot of model accuracies."""
    with open(experiment_path / "experiment.json", encoding="utf-8") as f:
        data = json.load(f)

    fig = make_subplots(rows=1, cols=2, subplot_titles=["Training Accuracy", "Test Accuracy"])

    # Plot training accuracies
    for model in ["quaternion", "tree", "logistic"]:
        fig.add_trace(
            go.Scatter(
                x=list(range(len(data[model]["train_accuracies"]))),
                y=data[model]["train_accuracies"],
                name=f"{model.capitalize()} (train)",
                line={"dash": "solid"},
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=list(range(len(data[model]["test_accuracies"]))),
                y=data[model]["test_accuracies"],
                name=f"{model.capitalize()} (test)",
                line={"dash": "dot"},
            ),
            row=1,
            col=2,
        )

    fig.update_layout(
        height=400,
        title="Model Accuracy Comparison",
        xaxis_title="Epoch",
        yaxis_title="Accuracy",
        hovermode="x unified",
        template="plotly_white",
    )

    return fig


def launch_dashboard() -> None:
    """Launch Streamlit dashboard with model comparisons."""
    st.set_page_config(page_title="Model Comparison Dashboard", layout="wide")
    st.title("Model Comparison Dashboard")

    # Experiment selection
    experiments_dir = Path("data/experiments")
    experiments = sorted([d for d in experiments_dir.iterdir() if d.is_dir()], reverse=True)

    if not experiments:
        st.error("No experiments found in data/experiments directory")
        return

    selected_exp = st.selectbox(
        "Select experiment:",
        experiments,
        format_func=lambda x: x.name,
    )

    # Plot accuracy comparison
    acc_fig = plot_accuracy_comparison(selected_exp)
    st.plotly_chart(acc_fig, use_container_width=True)

    # Plot quaternion weight evolution
    weight_fig = plot_weight_evolution(selected_exp)
    st.plotly_chart(weight_fig, use_container_width=True)

    # Display final metrics
    with open(selected_exp / "experiment.json", encoding="utf-8") as f:
        data = json.load(f)

    cols = st.columns(3)
    for i, model in enumerate(["quaternion", "tree", "logistic"]):
        with cols[i]:
            st.metric(
                f"{model.capitalize()} Final Test Accuracy",
                f"{data[model]['test_accuracies'][-1]:.4f}",
            )


if __name__ == "__main__":
    launch_dashboard()
