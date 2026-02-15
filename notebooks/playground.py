import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import numpy as np

    return Path, np


@app.cell
def _(np):
    rng = np.random.default_rng(seed=1)
    return


@app.cell
def _(Path, np):
    data_dir = Path("data", "binary_3sphere")
    train = np.load(data_dir / "train.npz")
    test = np.load(data_dir / "test.npz")
    print(f"Train X shape: {train['X'].shape}, y shape: {train['y'].shape}")
    print(f"Test  X shape: {test['X'].shape}, y shape: {test['y'].shape}")

    for i in range(3):
        print(f"  X = {train['X'][i]}, y = {train['y'][i]}")
    print("---")
    for i in range(3):
        print(f"  X = {test['X'][i]}, y = {test['y'][i]}")

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
