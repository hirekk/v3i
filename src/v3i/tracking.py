import json
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


def plot_action_evolution(experiment_path: Path) -> go.Figure:
    """Create an interactive plot of weight component evolution."""
    with Path(experiment_path / "experiment.json").open(encoding="utf-8") as f:
        data = json.load(f)
    # Create figure
    fig = go.Figure()

    if "quaternion" in data:
        plot_quaternion_weight_evolution(data["quaternion"]["action_history"], fig)
    if "octonion" in data:
        plot_octonion_weight_evolution(data["octonion"]["action_history"], fig)

    # Update layout
    fig.update_layout(
        title="Action Components Evolution",
        xaxis_title="Epoch",
        yaxis_title="Component Value",
        hovermode="x unified",
        template="plotly_white",
        showlegend=True,
        legend={
            "yanchor": "middle",
            "y": 0.5,
            "xanchor": "left",
            "x": 1.02,
            "orientation": "v",
        },
        margin={"r": 150},  # Add right margin for legend
    )

    return fig


def plot_bias_evolution(experiment_path: Path) -> go.Figure:
    """Create an interactive plot of weight component evolution."""
    with Path(experiment_path / "experiment.json").open(encoding="utf-8") as f:
        data = json.load(f)
    # Create figure
    fig = go.Figure()

    if "quaternion" in data:
        plot_quaternion_weight_evolution(data["quaternion"]["bias_history"], fig)
    if "octonion" in data:
        plot_octonion_weight_evolution(data["octonion"]["bias_history"], fig)

    # Update layout
    fig.update_layout(
        title="Bias Components Evolution",
        xaxis_title="Epoch",
        yaxis_title="Component Value",
        hovermode="x unified",
        template="plotly_white",
        showlegend=True,
        legend={
            "yanchor": "middle",
            "y": 0.5,
            "xanchor": "left",
            "x": 1.02,
            "orientation": "v",
        },
        margin={"r": 150},  # Add right margin for legend
    )

    return fig


def plot_quaternion_weight_evolution(weight_history: list[dict], figure: go.Figure) -> go.Figure:
    """Create an interactive plot of weight component evolution."""
    # Extract weight history
    steps = []
    w_comp = []
    x_comp = []
    y_comp = []
    z_comp = []

    for record in weight_history:
        steps.append(record["epoch"] + record["step"] / 10_000)  # normalize step to epoch
        w_comp.append(record["w"])
        x_comp.append(record["x"])
        y_comp.append(record["y"])
        z_comp.append(record["z"])

    # Add traces once for each component
    components = {
        "w (real)": (w_comp, "blue"),
        "x (i)": (x_comp, "red"),
        "y (j)": (y_comp, "green"),
        "z (k)": (z_comp, "purple"),
    }

    for name, (values, color) in components.items():
        figure.add_trace(
            go.Scatter(
                x=steps,
                y=values,
                mode="lines",
                name=name,
                line={"color": color},
            ),
        )

    return figure


def plot_octonion_weight_evolution(weight_history: list[dict], figure: go.Figure) -> go.Figure:
    """Create an interactive plot of weight component evolution."""
    # Extract weight history
    steps = []
    components = {f"x{i}": [] for i in range(8)}

    # Collect all data points first
    for record in weight_history:
        steps.append(record["epoch"] + record["step"] / 10_000)
        for i in range(8):
            components[f"x{i}"].append(record[f"x{i}"])

    # Colors for each component
    colors = ["blue", "red", "green", "purple", "orange", "brown", "pink", "gray"]

    # Add one trace per component
    for (name, values), color in zip(components.items(), colors, strict=True):
        figure.add_trace(
            go.Scatter(
                x=steps,
                y=values,
                mode="lines",
                name=name,
                line={"color": color},
            ),
        )

    return figure


def plot_accuracy_comparison(experiment_path: Path) -> go.Figure:
    """Create comparison plot showing accuracy evolution for all models."""
    with Path(experiment_path / "experiment.json").open(encoding="utf-8") as f:
        data = json.load(f)

    # Create figure with two subplots side by side
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Training Accuracy", "Test Accuracy"],
        horizontal_spacing=0.1,
    )

    # Colors and display names for different models
    model_configs = {
        "quaternion": {"color": "blue", "display": "Quaternion Perceptron"},
        "octonion": {"color": "purple", "display": "Octonion Perceptron"},
        "decision_tree": {"color": "red", "display": "Decision Tree"},
        "logistic": {"color": "green", "display": "Logistic Regression"},
        "random": {"color": "gray", "display": "Random Baseline"},
    }

    # Plot each model's accuracies
    for model_name, config in model_configs.items():
        if model_name not in data:
            continue

        # Training accuracy
        fig.add_trace(
            go.Scatter(
                x=list(range(len(data[model_name]["train_accuracies"]))),
                y=[
                    acc * 100 for acc in data[model_name]["train_accuracies"]
                ],  # Convert to percentage
                name=f"{config['display']} (train)",
                line={
                    "color": config["color"],
                    "width": 2,
                },
            ),
            row=1,
            col=1,
        )

        # Test accuracy
        fig.add_trace(
            go.Scatter(
                x=list(range(len(data[model_name]["test_accuracies"]))),
                y=[
                    acc * 100 for acc in data[model_name]["test_accuracies"]
                ],  # Convert to percentage
                name=f"{config['display']} (test)",
                line={
                    "color": config["color"],
                    "width": 2,
                    "dash": "dot",  # Dotted line for test accuracies
                },
            ),
            row=1,
            col=2,
        )

    # Update layout
    fig.update_layout(
        height=400,
        title="Model Accuracy Comparison",
        template="plotly_white",
        showlegend=True,
        legend={
            "yanchor": "middle",
            "y": 0.5,
            "xanchor": "left",
            "x": 1.02,  # Move legend to right side
            "orientation": "v",
        },
        margin={"r": 150},  # Add right margin for legend
        hovermode="x unified",
    )

    # Update axes
    for i in range(1, 3):
        fig.update_xaxes(
            title_text="Epoch",
            row=1,
            col=i,
            gridcolor="lightgray",
        )
        fig.update_yaxes(
            title_text="Accuracy (%)",
            row=1,
            col=i,
            range=[0, 100],  # Fix y-axis range from 0 to 100%
            gridcolor="lightgray",
            tickformat=".1f",  # Show as percentages without decimal places
        )

    return fig


def plot_metrics_change(experiment_path: Path) -> go.Figure:
    """Create a plot of metrics change."""
    # Display final metrics
    with Path(experiment_path / "experiment.json").open(encoding="utf-8") as f:
        data = json.load(f)

    # Create columns for final metrics
    cols = st.columns(len(data))
    for i, (model_name, model_data) in enumerate(data.items()):
        with cols[i]:
            display_name = {
                "quaternion": "Quaternion Perceptron",
                "decision_tree": "Decision Tree",
                "logistic": "Logistic Regression",
                "random": "Random Baseline",
            }.get(model_name, model_name.title())

            st.metric(
                f"{display_name}",
                f"{model_data['test_accuracies'][-1]:.1%}",
                f"{model_data['test_accuracies'][-1] - model_data['test_accuracies'][0]:.1%}",
                help="Final test accuracy (change from initial)",
            )


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

    # Metrics changeplot
    plot_metrics_change(selected_exp)
    # st.plotly_chart(acc_fig, use_container_width=True)

    # Plot weight evolution
    action_fig = plot_action_evolution(selected_exp)
    st.plotly_chart(action_fig, use_container_width=True)

    bias_fig = plot_bias_evolution(selected_exp)
    st.plotly_chart(bias_fig, use_container_width=True)


if __name__ == "__main__":
    launch_dashboard()
