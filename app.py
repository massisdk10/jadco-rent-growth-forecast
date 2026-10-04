"""Local Streamlit dashboard for the fixed JADCO 50/50 backup ensemble."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.features.model_dataset import SOURCE_FEATURES
from src.models.ensemble_50_50 import (
    ENSEMBLE_WEIGHTS,
    backtest_ensemble_50_50,
    estimate_2026_ensemble,
)


ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
DATASET_NAMES = (
    "equinoxe_lease_history.csv",
    "equinoxe_listings.csv",
    "equinoxe_concessions.csv",
    "equinoxe_asking_history.csv",
)
KEY_COLUMNS = ("sPropCode", "sUnitCode", "sBuilding", "sState")
REQUIRED_LEASE_COLUMNS = frozenset(
    {
        *KEY_COLUMNS,
        *SOURCE_FEATURES.values(),
        "sLeaseFrom",
        "sLeaseTo",
        "sSignDate",
        "sRentEffective",
    }
)


def _read_csv(payload: bytes, filename: str) -> pd.DataFrame:
    """Parse a CSV in memory while preserving identifiers as strings."""
    if not filename.lower().endswith(".csv"):
        raise ValueError("Le fichier doit être au format CSV.")
    return pd.read_csv(
        BytesIO(payload),
        dtype={column: "string" for column in KEY_COLUMNS},
    )


def _load_local_csv(filename: str) -> tuple[pd.DataFrame | None, str | None]:
    path = RAW_DIR / filename
    if not path.is_file():
        return None, None
    try:
        return _read_csv(path.read_bytes(), filename), None
    except (
        OSError,
        UnicodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        ValueError,
    ) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _dataset_status(
    filename: str, frame: pd.DataFrame | None, error: str | None
) -> dict[str, object]:
    if error:
        return {
            "file": filename,
            "rows": None,
            "columns": None,
            "status": "Invalid",
        }
    if frame is None:
        return {
            "file": filename,
            "rows": None,
            "columns": None,
            "status": "Missing",
        }
    return {
        "file": filename,
        "rows": len(frame),
        "columns": len(frame.columns),
        "status": "Loaded",
    }


def _validate_lease_history(
    frame: pd.DataFrame | None, error: str | None
) -> tuple[bool, list[str]]:
    if error:
        return False, [error]
    if frame is None:
        return False, ["equinoxe_lease_history.csv is required."]
    missing = sorted(REQUIRED_LEASE_COLUMNS - set(frame.columns))
    return not missing, missing


def _visible_error(error: str) -> str:
    """Avoid surfacing uploaded values or CSV parser excerpts to the UI."""
    return error.split(":", maxsplit=1)[0]


def _json_download(
    forecast: dict[str, Any], metrics: pd.DataFrame
) -> bytes:
    """Serialize only portfolio values and aggregate metrics."""
    metric_columns = (
        "model",
        "MAE",
        "RMSE",
        "bias",
        "worst_absolute_error",
    )
    if not set(metric_columns).issubset(metrics.columns):
        raise ValueError("Aggregate metrics table has an unexpected schema.")
    payload = {
        "forecast_year": int(forecast["forecast_year"]),
        "model_a": float(forecast["model_a"]),
        "model_b": float(forecast["model_b"]),
        "ensemble_50_50": float(forecast["ensemble_50_50"]),
        "weights": dict(ENSEMBLE_WEIGHTS),
        "aggregate_backtest_metrics": metrics.loc[
            :, list(metric_columns)
        ].to_dict(orient="records"),
    }
    return json.dumps(payload, indent=2, allow_nan=False).encode("utf-8")


st.set_page_config(
    page_title="JADCO / Collection Équinoxe",
    page_icon="🏠",
    layout="wide",
)
st.title("JADCO / Collection Équinoxe")
st.subheader("2026 Rent Growth Forecast — Backup Ensemble")
st.caption("Fixed 50/50 Ensemble | Model A + Model B")

with st.sidebar:
    st.header("Datasets")
    source = st.radio(
        "Dataset source",
        ("Local project files", "Upload CSVs"),
        help="Les fichiers sont lus en mémoire uniquement et ne sont jamais écrits dans le dépôt.",
    )
    frames: dict[str, pd.DataFrame | None] = {}
    errors: dict[str, str | None] = {}
    for filename in DATASET_NAMES:
        if source == "Local project files":
            frame, error = _load_local_csv(filename)
        else:
            uploaded = st.file_uploader(
                filename,
                type=["csv"],
                key=f"upload_{filename}",
                help="Le contenu ne sera ni sauvegardé, ni affiché ligne par ligne.",
            )
            if uploaded is None:
                frame, error = None, None
            else:
                try:
                    frame = _read_csv(uploaded.getvalue(), uploaded.name)
                    error = None
                except (
                    OSError,
                    UnicodeError,
                    pd.errors.EmptyDataError,
                    pd.errors.ParserError,
                    ValueError,
                ) as exc:
                    frame, error = None, f"{type(exc).__name__}: {exc}"
        frames[filename] = frame
        errors[filename] = error

    dataset_summary = pd.DataFrame(
        [
            _dataset_status(filename, frames[filename], errors[filename])
            for filename in DATASET_NAMES
        ]
    )
    st.dataframe(dataset_summary, hide_index=True, width="stretch")
    st.caption(
        "Le forecast utilise uniquement lease history. Listings, concessions et "
        "asking sont acceptés pour compatibilité mais ne sont pas des entrées prédictives."
    )

lease_filename = "equinoxe_lease_history.csv"
lease_history = frames[lease_filename]
is_valid, validation_issues = _validate_lease_history(
    lease_history, errors[lease_filename]
)
st.header("Overview")
if is_valid:
    st.success("✅ Dataset valid")
else:
    issue = validation_issues[0]
    if issue.startswith("Missing"):
        st.error(f"❌ {issue}")
    elif errors[lease_filename] is not None:
        st.error(f"❌ CSV invalid: {_visible_error(issue)}")
    else:
        st.error(f"❌ Missing required columns: {', '.join(validation_issues)}")

if source == "Upload CSVs":
    st.info(
        "Les fichiers restent en mémoire pendant cette session Streamlit. "
        "Aucune table CRM n’est affichée."
    )

if "forecast_result" not in st.session_state:
    st.session_state["forecast_result"] = None
    st.session_state["backtest_rows"] = None
    st.session_state["backtest_metrics"] = None

st.header("2026 Forecast")
run_forecast = st.button(
    "Run 2026 Forecast",
    type="primary",
    disabled=not is_valid,
    help="Prévision cutoff-safe avec les APIs Model A, Model B P1_G1 et le wrapper 50/50.",
)
if run_forecast and is_valid and lease_history is not None:
    with st.spinner("Calcul des folds et du forecast 2026…"):
        try:
            asking = frames["equinoxe_asking_history.csv"]
            fold_rows, metrics = backtest_ensemble_50_50(lease_history)
            forecast = estimate_2026_ensemble(lease_history, asking=asking)
            exact_average = (
                ENSEMBLE_WEIGHTS["model_a"] * forecast["model_a"]
                + ENSEMBLE_WEIGHTS["model_b"] * forecast["model_b"]
            )
            if forecast["ensemble_50_50"] != exact_average:
                raise AssertionError("The fixed 50/50 ensemble arithmetic failed.")
            st.session_state["forecast_result"] = forecast
            st.session_state["backtest_rows"] = fold_rows
            st.session_state["backtest_metrics"] = metrics
        except Exception as exc:
            st.session_state["forecast_result"] = None
            st.session_state["backtest_rows"] = None
            st.session_state["backtest_metrics"] = None
            st.error(
                f"Forecast failed ({type(exc).__name__}). "
                "Vérifie la validité et la complétude du dataset local."
            )

forecast_result = st.session_state["forecast_result"]
if forecast_result is not None:
    model_a_col, model_b_col, ensemble_col = st.columns(3)
    model_a_col.metric("MODEL A", f"{forecast_result['model_a']:.6f} %")
    model_b_col.metric("MODEL B P1_G1", f"{forecast_result['model_b']:.6f} %")
    ensemble_col.metric(
        "Backup Forecast 2026",
        f"{forecast_result['ensemble_50_50']:.6f} %",
        help="Exactement 0.50 × Model A + 0.50 × Model B.",
    )

st.header("Historical Backtest")
fold_rows = st.session_state["backtest_rows"]
backtest_metrics = st.session_state["backtest_metrics"]
if fold_rows is None or backtest_metrics is None:
    st.info("Lance le forecast pour calculer et afficher les folds 2023–2025.")
else:
    st.dataframe(fold_rows, hide_index=True, width="stretch")
    st.dataframe(backtest_metrics, hide_index=True, width="stretch")
    chart = fold_rows.set_index("year")[
        ["realized", "A_prediction", "B_prediction", "ensemble_50_50"]
    ].rename(
        columns={
            "realized": "Actual",
            "A_prediction": "Model A",
            "B_prediction": "Model B",
            "ensemble_50_50": "Ensemble",
        }
    )
    st.line_chart(chart, y_label="P1 (%)", x_label="Year")

    if forecast_result is not None:
        st.download_button(
            "Download Results",
            data=_json_download(forecast_result, backtest_metrics),
            file_name="jadco_backup_ensemble_aggregated.json",
            mime="application/json",
            help="Export portefeuille uniquement : forecasts et métriques agrégées, aucun record CRM.",
        )

st.header("How it works")
st.markdown(
    """
**Target:** Indice médian de contribution attendue des transitions à la croissance
annualisée du loyer effectif des unités à échéance.

- `p_i` = probability of lease transition
- `g_i` = expected annualized effective-rent growth if transition occurs
- `c_i = p_i × g_i`
- `P1 = median(c_i)`

**Model A** = approche de prévision historique globale et conservatrice.

**Model B P1_G1** = occurrence hiérarchique province → province + expiry month
et croissance hiérarchique par province.

**Backup ensemble** = 50 % Model A + 50 % Model B.

**The 50/50 weights are fixed and were NOT optimized on historical backtests.**
"""
)

with st.expander("Leakage Safety"):
    st.markdown(
        """
        - 2023 → cutoff 2022-12-31
        - 2024 → cutoff 2023-12-31
        - 2025 → cutoff 2024-12-31
        - 2026 → cutoff 2025-12-31

        **No information after each prediction origin is allowed.** Les cohortes,
        labels qualifiés, dates de maturité et variables sont préparés par les APIs
        existantes ; le dashboard ne remplace aucune règle anti-leakage.
        """
    )

with st.expander("Limitations"):
    st.markdown(
        """
        - Only three historical backtest years.
        - P1 is an event-based contribution index.
        - P1 is **not** total rent-roll growth.
        - Occurrence regime changed materially in 2025.
        - Model A and Model B have distinct assumptions.
        - 50/50 is a neutral fallback, not an optimized ensemble.
        - CRM data is confidential and must remain local.
        - Listings, concessions and asking history are not used by this final
          forecast; they are accepted only for compatibility.
        """
    )
