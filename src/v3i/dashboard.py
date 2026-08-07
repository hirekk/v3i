"""Streamlit dashboard comparing experiment runs from runs/*.json.

Panels: test/train accuracy per epoch (with the proven 75% XOR linear ceiling
and baseline reference lines), geodesic loss per epoch, and weight-component
evolution for one selected run/layer. Runs are produced by v3i.run_experiment.

Launch:
    uv run streamlit run src/v3i/dashboard.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from v3i import xor_geometry as xg

RUNS_DIR = Path("runs")

# Validated categorical palette (dataviz reference, light mode, fixed order).
SERIES = ["#2a78d6", "#008300", "#e87ba4", "#eda100", "#1baf7a", "#eb6834", "#4a3aa7", "#e34948"]
SURFACE, INK, INK_2, MUTED, GRID, AXIS = (
    "#fcfcfb",
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#c3c2b7",
)
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def base_layout(title: str, y_title: str) -> go.Layout:
    """Shared chart chrome: surface, ink, grid, unified hover."""
    return go.Layout(
        title={"text": title, "font": {"size": 15, "color": INK}},
        font={"family": FONT, "size": 12, "color": INK_2},
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        xaxis={
            "title": "epoch",
            "gridcolor": GRID,
            "linecolor": AXIS,
            "zeroline": False,
            "tickcolor": AXIS,
            "tickfont": {"color": MUTED},
        },
        yaxis={
            "title": y_title,
            "gridcolor": GRID,
            "linecolor": AXIS,
            "zeroline": False,
            "tickcolor": AXIS,
            "tickfont": {"color": MUTED},
        },
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        margin={"l": 56, "r": 24, "t": 56, "b": 44},
        height=380,
    )


@st.cache_data
def load_runs(mtimes: tuple[tuple[str, float], ...]) -> dict[str, dict]:
    """Load run JSONs, cached on (path, mtime) so edits invalidate."""
    return {Path(name).stem: json.loads(Path(name).read_text()) for name, _ in mtimes}


def discover() -> dict[str, dict]:
    """Discover all runs under runs/."""
    files = sorted(RUNS_DIR.glob("*.json"))
    return load_runs(tuple((str(f), f.stat().st_mtime) for f in files))


def metric_traces(
    runs: dict[str, dict], selected: list[str], hue: dict[str, str], key: str
) -> list[go.Scatter]:
    """Solid test / dashed train traces for one metric family, hue per run."""
    traces = []
    for name in selected:
        metrics = runs[name].get("metrics")
        if not metrics:
            continue
        epochs = [m["epoch"] for m in metrics]
        traces.append(
            go.Scatter(
                x=epochs,
                y=[m[f"test_{key}"] for m in metrics],
                name=name,
                mode="lines",
                line={"color": hue[name], "width": 2},
                legendgroup=name,
            )
        )
        traces.append(
            go.Scatter(
                x=epochs,
                y=[m[f"train_{key}"] for m in metrics],
                name=f"{name} (train)",
                mode="lines",
                line={"color": hue[name], "width": 2, "dash": "dot"},
                legendgroup=name,
                showlegend=False,
                opacity=0.55,
            )
        )
    return traces


def reference_lines(fig: go.Figure, runs: dict[str, dict], selected: list[str]) -> None:
    """75% ceiling for XOR runs and baseline test accuracies, as recessive refs."""
    if any(runs[n]["config"]["dataset"] == "binary-xor" for n in selected):
        fig.add_hline(
            y=0.75,
            line={"color": MUTED, "dash": "dash", "width": 1},
            annotation_text="75% linear ceiling (proven)",
            annotation_font_color=MUTED,
        )
    for run in runs.values():
        for b in run.get("baselines", []):
            fig.add_hline(
                y=b["test_acc"],
                line={"color": AXIS, "dash": "dot", "width": 1},
                annotation_text=f"{b['model']} {b['test_acc']:.2f}",
                annotation_position="bottom right",
                annotation_font_color=MUTED,
            )


def training_view() -> None:
    """Compare training runs: accuracy, geodesic loss, weight evolution."""
    runs = discover()
    trained = [n for n, r in runs.items() if r.get("metrics")]
    if not trained:
        st.info(
            "No runs found. Create some with: "
            "`uv run python -m v3i.run_experiment --dataset binary-xor --model octonion`"
        )
        return

    with st.sidebar:
        st.header("Runs")
        selected = st.multiselect(
            "Compare runs", trained, default=trained[:4], max_selections=len(SERIES)
        )
        datasets = {runs[n]["config"]["dataset"] for n in selected}
        if len(datasets) > 1:
            st.warning(f"Mixed datasets selected: {', '.join(sorted(datasets))}")
        st.caption(
            "Solid line = test, dotted = train. Reference lines are baselines fit on the same data."
        )

    if not selected:
        st.info("Select at least one run.")
        return

    # Hue follows the run's position in the stable discovered list, not the
    # selection, so deselecting one run never repaints the others.
    hue = {name: SERIES[i % len(SERIES)] for i, name in enumerate(trained)}

    tiles = st.columns(len(selected))
    for tile, name in zip(tiles, selected, strict=True):
        final = runs[name]["metrics"][-1]
        is_xor = runs[name]["config"]["dataset"] == "binary-xor"
        tile.metric(
            label=name,
            value=f"{final['test_acc']:.1%}",
            delta=f"{final['test_acc'] - 0.75:+.1%} vs ceiling" if is_xor else None,
        )

    acc = go.Figure(layout=base_layout("Accuracy", "accuracy"))
    for t in metric_traces(runs, selected, hue, "acc"):
        acc.add_trace(t)
    reference_lines(acc, runs, selected)
    acc.update_yaxes(range=[0, 1.02], tickformat=".0%")
    st.plotly_chart(acc, width="stretch")

    loss = go.Figure(layout=base_layout("Geodesic loss (mean angle to target pole)", "radians"))
    for t in metric_traces(runs, selected, hue, "loss"):
        loss.add_trace(t)
    loss.add_hline(
        y=np.pi / 2,
        line={"color": MUTED, "dash": "dash", "width": 1},
        annotation_text="π/2 = orthogonal (chance)",
        annotation_font_color=MUTED,
    )
    st.plotly_chart(loss, width="stretch")

    st.subheader("Weight evolution")
    col_run, col_layer = st.columns(2)
    w_run = col_run.selectbox("Run", selected)
    n_layers = len(runs[w_run]["metrics"][0]["weights"])
    layer = col_layer.selectbox("Layer", range(n_layers)) if n_layers > 1 else 0

    snaps = np.array([m["weights"][layer] for m in runs[w_run]["metrics"]])
    epochs = [m["epoch"] for m in runs[w_run]["metrics"]]
    dim = snaps.shape[1]
    comp = go.Figure(layout=base_layout(f"{w_run} — layer {layer} weight components", "value"))
    for i in range(dim):
        comp.add_trace(
            go.Scatter(
                x=epochs,
                y=snaps[:, i],
                name=f"e{i}" if i else "re",
                mode="lines",
                line={"color": SERIES[i % len(SERIES)], "width": 2},
            )
        )
    st.plotly_chart(comp, width="stretch")

    angle = go.Figure(layout=base_layout("Angle of weight to identity", "radians"))
    angle.add_trace(
        go.Scatter(
            x=epochs,
            y=np.arccos(np.clip(snaps[:, 0], -1, 1)),
            mode="lines",
            line={"color": SERIES[0], "width": 2},
            showlegend=False,
        )
    )
    st.plotly_chart(angle, width="stretch")

    with st.expander("Table view (all selected runs, per epoch)"):
        rows = [
            {"run": name, **{k: v for k, v in m.items() if k != "weights"}}
            for name in selected
            for m in runs[name]["metrics"]
        ]
        st.dataframe(rows, width="stretch")


# Diverging accuracy scale centred on the 75% linear ceiling: blue below,
# neutral at the ceiling, red above (a red point is a genuine break-through).
_ACC_SCALE = [[0.0, "#2a78d6"], [0.5, "#d9d7cf"], [1.0, "#e34948"]]


@st.cache_data
def _xor_data(seed: int, noise: float) -> dict[str, np.ndarray]:
    return xg.load_xor(seed=seed, noise=noise)


@st.cache_data
def _linear_sphere(seed: int, noise: float, n_draws: int) -> dict[str, np.ndarray]:
    return xg.sample_linear_separators(_xor_data(seed, noise), n_draws=n_draws, seed=0)


@st.cache_data
def _mech_acc(seed: int, noise: float, mechanism: str, n_draws: int) -> np.ndarray:
    return xg.sample_mechanism_accuracies(
        _xor_data(seed, noise), mechanism, n_draws=n_draws, seed=0
    )


def xor_geometry_view() -> None:
    """The XOR dataset and the space of weights that discriminate it."""
    with st.sidebar:
        st.header("XOR geometry")
        seed = st.number_input("Data seed", value=42, step=1)
        noise = st.slider("Blob noise", 0.02, 0.30, 0.10, 0.02)
        n_draws = st.select_slider("Random weight draws", [1000, 2000, 4000, 8000], value=4000)
        st.caption(
            "Random-weight *screen*: shows which mechanisms **can** exceed the "
            "75% linear ceiling, not whether the error wave can learn to."
        )

    data = _xor_data(seed, noise)

    st.subheader("The dataset")
    st.caption(
        "XOR is four blobs in the plane, labelled ±1 on the diagonal — not "
        "linearly separable. Inverse-stereographic embedding sends it to a "
        "2-sphere inside R⁸ (coords 0,1,2; the other five stay zero)."
    )
    plane = go.Figure(layout=base_layout("XOR in the source plane", "y"))
    for label, colour in [(-1, SERIES[0]), (1, SERIES[7])]:
        mask = data["y_all"] == label
        plane.add_trace(
            go.Scatter(
                x=data["plane"][mask, 0],
                y=data["plane"][mask, 1],
                mode="markers",
                name=f"class {label:+d}",
                marker={
                    "color": colour,
                    "size": 5,
                    "opacity": 0.6,
                    "line": {"color": SURFACE, "width": 0.5},
                },
            )
        )
    plane.update_xaxes(title="x")
    st.plotly_chart(plane, width="stretch")

    st.subheader("Weights that discriminate — the linear readout")
    st.caption(
        "For `sign(re(x·w))` on this data, only the separator normal "
        "`(w₀, -w₁, -w₂)` matters — a point on an ordinary 2-sphere. Each dot "
        "is a random weight, coloured by its XOR accuracy. The ceiling is "
        "**geometric**: no direction is red."
    )
    sph = _linear_sphere(seed, noise, n_draws)
    best = float(sph["acc"].max())
    fig3d = go.Figure(
        go.Scatter3d(
            x=sph["normals"][:, 0],
            y=sph["normals"][:, 1],
            z=sph["normals"][:, 2],
            mode="markers",
            marker={
                "size": 2.5,
                "color": sph["acc"],
                "colorscale": _ACC_SCALE,
                "cmin": 0.5,
                "cmax": 1.0,
                "opacity": 0.7,
                "colorbar": {"title": "acc", "tickformat": ".0%"},
            },
            hovertemplate="acc %{marker.color:.1%}<extra></extra>",
        )
    )
    fig3d.update_layout(
        paper_bgcolor=SURFACE,
        font={"family": FONT, "color": INK_2},
        height=460,
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        scene={
            "xaxis_title": "w₀",
            "yaxis_title": "-w₁",
            "zaxis_title": "-w₂",
            "aspectmode": "cube",
        },
    )
    st.plotly_chart(fig3d, width="stretch")
    st.metric(
        "Best linear separator (of the draws)",
        f"{best:.1%}",
        delta=f"{best - xg.XOR_LINEAR_CEILING:+.1%} vs ceiling",
    )

    st.subheader("Do richer combiners break through?")
    st.caption(
        "Each combiner uses several weights, so it has no single 2-sphere — "
        "compare the *distribution* of accuracy over random draws. Mass to the "
        "right of the dashed line is a break-through the linear readout cannot make."
    )
    dist = go.Figure(layout=base_layout("Accuracy distribution over random draws", "count"))
    summary = []
    for i, mech in enumerate(xg.MECHANISMS):
        acc = _mech_acc(seed, noise, mech, min(n_draws, 4000))
        dist.add_trace(
            go.Histogram(
                x=acc,
                name=mech,
                marker_color=SERIES[i],
                opacity=0.55,
                nbinsx=48,
            )
        )
        summary.append(
            {
                "mechanism": mech,
                "best": f"{acc.max():.1%}",
                "median": f"{np.median(acc):.1%}",
                "frac > 75%": f"{np.mean(acc > xg.XOR_LINEAR_CEILING):.1%}",
            }
        )
    dist.update_layout(barmode="overlay")
    dist.add_vline(
        x=xg.XOR_LINEAR_CEILING,
        line={"color": MUTED, "dash": "dash", "width": 1},
        annotation_text="75% ceiling",
        annotation_font_color=MUTED,
    )
    dist.update_xaxes(title="best-orientation accuracy", tickformat=".0%")
    st.plotly_chart(dist, width="stretch")
    st.dataframe(summary, width="stretch")


def main() -> None:
    """Dashboard entry point: pick a view."""
    st.set_page_config(page_title="v3i experiments", layout="wide")
    st.title("v3i — hypercomplex perceptron experiments")
    view = st.sidebar.radio("View", ["Training comparison", "XOR geometry"])
    if view == "Training comparison":
        training_view()
    else:
        xor_geometry_view()


main()
