"""Data preparation utilities for AI4I 2020 Predictive Maintenance dataset.

This module provides:
1) Ingestion and cleaning utilities for the Kaggle AI4I 2020 dataset.
2) Standardized column naming and missing-value handling.
3) Manufacturing KPI functions for OEE, MTBF, and MTTR.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class ProductionTimeInput:
    """Container for production-time metrics used for OEE calculation.

    Attributes:
        planned_production_time_min: Planned production time (minutes).
        downtime_min: Unplanned downtime (minutes).
        ideal_cycle_time_min_per_unit: Ideal cycle time per produced unit (minutes/unit).
        total_units: Total produced units.
        good_units: Conforming (non-defective) units.
    """

    planned_production_time_min: float = 480.0
    downtime_min: float = 45.0
    ideal_cycle_time_min_per_unit: float = 0.50
    total_units: int = 700
    good_units: int = 672


@dataclass
class FailureRepairInput:
    """Container for reliability metrics used in MTBF/MTTR calculation."""

    total_uptime_hours: float = 500.0
    failure_count: int = 6
    total_repair_time_hours: float = 18.0


def standardize_column_name(column_name: str) -> str:
    """Convert raw dataset columns to snake_case names.

    Args:
        column_name: Original column name.

    Returns:
        Standardized snake_case name.
    """
    normalized = column_name.strip().lower()
    replacements = {
        "[k]": "_k",
        "[rpm]": "_rpm",
        "[nm]": "_nm",
        "[min]": "_min",
        " ": "_",
        "-": "_",
        "/": "_",
        "(": "",
        ")": "",
    }
    for src, target in replacements.items():
        normalized = normalized.replace(src, target)

    while "__" in normalized:
        normalized = normalized.replace("__", "_")

    return normalized.strip("_")


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of DataFrame with standardized column names."""
    renamed = df.copy()
    renamed.columns = [standardize_column_name(col) for col in renamed.columns]
    return renamed


def clean_ai4i_dataset(
    input_csv_path: Path | str,
    output_csv_path: Optional[Path | str] = None,
) -> pd.DataFrame:
    """Load, clean, and optionally persist AI4I dataset.

    Cleaning steps:
      - drop duplicate rows
      - standardize column names
      - infer and fill missing values (median for numeric; mode for categorical)
      - drop unnamed index-like columns

    Args:
        input_csv_path: Path to raw AI4I CSV file.
        output_csv_path: Optional path to save cleaned dataset.

    Returns:
        Cleaned pandas DataFrame.
    """
    input_csv_path = Path(input_csv_path)
    df = pd.read_csv(input_csv_path)

    df = df.drop_duplicates().copy()
    df = standardize_column_names(df)

    unnamed_columns = [c for c in df.columns if c.startswith("unnamed")]
    if unnamed_columns:
        df = df.drop(columns=unnamed_columns)

    for col in df.columns:
        if df[col].isna().sum() == 0:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            fill_value = df[col].median()
        else:
            mode_values = df[col].mode(dropna=True)
            fill_value = mode_values.iloc[0] if not mode_values.empty else "unknown"

        df[col] = df[col].fillna(fill_value)

    if output_csv_path:
        output_csv_path = Path(output_csv_path)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv_path, index=False)

    return df


def calculate_oee(metrics: ProductionTimeInput) -> Dict[str, float]:
    """Calculate OEE and components (Availability, Performance, Quality)."""
    operating_time = max(metrics.planned_production_time_min - metrics.downtime_min, 0.0)
    availability = operating_time / metrics.planned_production_time_min if metrics.planned_production_time_min > 0 else 0.0

    theoretical_output_time = metrics.ideal_cycle_time_min_per_unit * metrics.total_units
    performance = theoretical_output_time / operating_time if operating_time > 0 else 0.0
    performance = min(max(performance, 0.0), 1.0)

    quality = metrics.good_units / metrics.total_units if metrics.total_units > 0 else 0.0
    oee = availability * performance * quality

    return {
        "availability": availability,
        "performance": performance,
        "quality": quality,
        "oee": oee,
    }


def calculate_mtbf_mttr(metrics: FailureRepairInput) -> Dict[str, float]:
    """Calculate MTBF and MTTR reliability metrics."""
    failures = max(metrics.failure_count, 0)
    mtbf = metrics.total_uptime_hours / failures if failures > 0 else float("inf")
    mttr = metrics.total_repair_time_hours / failures if failures > 0 else 0.0

    return {"mtbf_hours": mtbf, "mttr_hours": mttr}


def compute_manufacturing_kpis(
    production_input: Optional[ProductionTimeInput] = None,
    repair_input: Optional[FailureRepairInput] = None,
) -> Dict[str, float]:
    """Compute and return a combined KPI dictionary for OEE, MTBF, and MTTR."""
    production_input = production_input or ProductionTimeInput()
    repair_input = repair_input or FailureRepairInput()

    oee_metrics = calculate_oee(production_input)
    reliability_metrics = calculate_mtbf_mttr(repair_input)

    return {**oee_metrics, **reliability_metrics}


if __name__ == "__main__":
    RAW_DATA_PATH = Path("data/ai4i2020.csv")
    CLEAN_DATA_PATH = Path("data/ai4i2020_clean.csv")

    if RAW_DATA_PATH.exists():
        cleaned = clean_ai4i_dataset(RAW_DATA_PATH, CLEAN_DATA_PATH)
        print(f"✅ Cleaned dataset saved to: {CLEAN_DATA_PATH} ({len(cleaned)} rows)")
    else:
        print(f"⚠️ Raw dataset not found at {RAW_DATA_PATH}. Place AI4I CSV and rerun.")

    kpis = compute_manufacturing_kpis()
    print("Sample KPI Snapshot:")
    for metric, value in kpis.items():
        print(f"  - {metric}: {value:.4f}" if np.isfinite(value) else f"  - {metric}: inf")
