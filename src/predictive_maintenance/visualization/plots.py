from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def save_figure(
    fig,
    output_path: Path,
) -> None:
    """
    Save a matplotlib figure.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_rul_distribution(
    df: pd.DataFrame,
    output_path: Path,
    target_column: str = "RUL",
) -> None:
    """
    Plot the distribution of RUL values.
    """

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.histplot(
        data=df,
        x=target_column,
        kde=True,
        ax=ax,
    )

    ax.set_title("RUL Distribution")

    ax.set_xlabel("Remaining Useful Life")

    ax.set_ylabel("Frequency")

    save_figure(
        fig,
        output_path,
    )


def plot_sensor_distribution(
    df: pd.DataFrame,
    sensor: str,
    output_path: Path,
) -> None:
    """
    Plot the distribution of one sensor.
    """

    if sensor not in df.columns:
        raise ValueError(f"Sensor not found: {sensor}")

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.histplot(
        data=df,
        x=sensor,
        kde=True,
        ax=ax,
    )

    ax.set_title(f"{sensor} Distribution")

    ax.set_xlabel(sensor)
    ax.set_ylabel("Frequency")

    save_figure(
        fig,
        output_path,
    )


def plot_sensor_trend(
    df: pd.DataFrame,
    unit: int,
    sensor: str,
    output_path: Path,
) -> None:
    """
    Plot the evolution of a sensor for one engine.
    """

    if sensor not in df.columns:
        raise ValueError(f"Sensor not found: {sensor}")

    engine = df[df["unit"] == unit].sort_values("cycle")

    if engine.empty:
        raise ValueError(f"Unit not found: {unit}")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        engine["cycle"],
        engine[sensor],
    )

    ax.set_title(f"{sensor} evolution - Unit {unit}")

    ax.set_xlabel("Cycle")
    ax.set_ylabel(sensor)

    ax.grid(True)

    save_figure(
        fig,
        output_path,
    )


def plot_rul_by_cycle(
    df: pd.DataFrame,
    unit: int,
    output_path: Path,
) -> None:
    """
    Plot RUL evolution for one engine.
    """

    engine = df[df["unit"] == unit].sort_values("cycle")

    if engine.empty:
        raise ValueError(f"Unit not found: {unit}")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        engine["cycle"],
        engine["RUL"],
    )

    ax.set_title(f"RUL evolution - Unit {unit}")

    ax.set_xlabel("Cycle")
    ax.set_ylabel("RUL")

    ax.grid(True)

    save_figure(
        fig,
        output_path,
    )


def plot_correlation_matrix(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Plot correlation matrix for numerical features.
    """

    numeric_df = df.select_dtypes(include="number")

    correlation = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(14, 10))

    sns.heatmap(
        correlation,
        cmap="coolwarm",
        center=0,
        ax=ax,
    )

    ax.set_title("Feature Correlation Matrix")

    save_figure(
        fig,
        output_path,
    )


def plot_actual_vs_predicted(
    y_true: pd.Series,
    y_pred: pd.Series,
    output_path: Path,
) -> None:
    """
    Plot actual RUL against predicted RUL.
    """

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(
        y_true,
        y_pred,
        alpha=0.5,
    )

    # Perfect prediction line
    min_value = min(
        y_true.min(),
        y_pred.min(),
    )

    max_value = max(
        y_true.max(),
        y_pred.max(),
    )

    ax.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--",
    )

    ax.set_title("Actual vs Predicted RUL")

    ax.set_xlabel("Actual RUL")

    ax.set_ylabel("Predicted RUL")

    ax.grid(True)

    save_figure(
        fig,
        output_path,
    )


def plot_residuals(
    y_true: pd.Series,
    y_pred: pd.Series,
    output_path: Path,
) -> None:
    """
    Plot prediction residuals.
    """

    residuals = y_true - y_pred

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(
        y_pred,
        residuals,
        alpha=0.5,
    )

    ax.axhline(
        0,
        linestyle="--",
    )

    ax.set_title("Prediction Residuals")

    ax.set_xlabel("Predicted RUL")

    ax.set_ylabel("Residual")

    ax.grid(True)

    save_figure(
        fig,
        output_path,
    )


def plot_model_comparison(
    results: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Compare models using MAE and RMSE.
    """

    required_columns = {
        "model",
        "MAE",
        "RMSE",
    }

    missing = required_columns - set(results.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    fig, ax = plt.subplots(figsize=(10, 6))

    x = range(len(results))

    width = 0.35

    ax.bar(
        [i - width / 2 for i in x],
        results["MAE"],
        width,
        label="MAE",
    )

    ax.bar(
        [i + width / 2 for i in x],
        results["RMSE"],
        width,
        label="RMSE",
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        results["model"],
        rotation=30,
        ha="right",
    )

    ax.set_title("Model Performance Comparison")

    ax.set_ylabel("Error")

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    save_figure(
        fig,
        output_path,
    )
