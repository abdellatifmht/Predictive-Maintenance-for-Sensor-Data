import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from predictive_maintenance.config import RESULTS_DIR
from predictive_maintenance.monitoring.drift import (
    PSI_THRESHOLD_ALERT,
    PSI_THRESHOLD_WARNING,
    compare_all_features,
)

LOG_PATH = RESULTS_DIR / "monitoring" / "logs" / "predictions.jsonl"
REFERENCE_PATH = RESULTS_DIR / "monitoring" / "reference_profile.json"

st.set_page_config(page_title="Predictive Maintenance — Monitoring", layout="wide")


@st.cache_data(ttl=5)
def load_logs() -> pd.DataFrame:
    if not LOG_PATH.exists():
        return pd.DataFrame()

    rows = []
    with open(LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # A writer may still be appending the final line.

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    reading_cols = pd.json_normalize(df["last_reading"]).add_prefix("reading_")
    df = pd.concat([df.reset_index(drop=True), reading_cols.reset_index(drop=True)], axis=1)
    return df


@st.cache_data
def load_reference():
    if not REFERENCE_PATH.exists():
        return None
    with open(REFERENCE_PATH) as f:
        return json.load(f)


st.title("🔧 Predictive Maintenance — Dashboard de monitoring")

df = load_logs()
reference = load_reference()

if df.empty:
    st.warning(
        "Aucune donnée de monitoring pour l'instant. Démarre l'API (`make api`) puis "
        "lance `make monitoring-stream` pour générer du trafic, ensuite rafraîchis cette page."
    )
    st.stop()

if reference is None:
    st.error(
        "Profil de référence introuvable. Lance `make monitoring-reference` avant de continuer."
    )
    st.stop()

FEATURES = list(reference["features"].keys())

st.sidebar.header("Paramètres")
window_size = st.sidebar.slider(
    "Taille de la fenêtre de drift (dernières N requêtes)",
    min_value=10,
    max_value=max(len(df), 20),
    value=max(10, min(50, len(df))),
)
st.sidebar.metric("Requêtes totales reçues", len(df))
st.sidebar.metric("Requêtes en erreur", int((df["status"] != "ok").sum()))
if st.sidebar.button("🔄 Rafraîchir"):
    st.cache_data.clear()
    st.rerun()

current_window = df.tail(window_size)

reference_df = pd.DataFrame({f: reference["features"][f]["values"] for f in FEATURES})
current_df = current_window.rename(columns={f"reading_{f}": f for f in FEATURES})

drift_results = compare_all_features(reference_df, current_df, FEATURES)
n_alert = sum(1 for r in drift_results if r.psi_level == "alert")
n_warning = sum(1 for r in drift_results if r.psi_level == "warning")
any_drift = any(r.is_drifted for r in drift_results)

if any_drift:
    drifted_features = [r.feature for r in drift_results if r.is_drifted]
    st.error(
        f"🚨 **Drift détecté** sur {len(drifted_features)} capteur(s) : {', '.join(drifted_features)} "
        f"— sur les {window_size} dernières requêtes."
    )
elif n_warning > 0:
    st.warning(f"⚠️ {n_warning} capteur(s) en zone de surveillance.")
else:
    st.success(
        "✅ Aucun drift détecté — distributions stables par rapport à la référence (train)."
    )

col1, col2, col3, col4 = st.columns(4)
col1.metric(
    "RUL moyenne prédite (fenêtre)",
    f"{current_window['predicted_rul'].mean():.1f}" if "predicted_rul" in current_window else "NA",
)
col2.metric(
    "Latence moyenne (ms)",
    f"{current_window['latency_ms'].mean():.1f}" if "latency_ms" in current_window else "NA",
)
col3.metric("Capteurs en alerte", n_alert)
col4.metric("Capteurs en surveillance", n_warning)

st.subheader("Distribution des prédictions dans le temps")
tab1, tab2 = st.tabs(["RUL prédite", "Classe de santé"])

with tab1:
    if "predicted_rul" in df.columns:
        fig = px.scatter(
            df,
            x="timestamp",
            y="predicted_rul",
            color="drift_injected",
            color_discrete_map={True: "red", False: "steelblue"},
            title="Predicted RUL over time (red = simulated drift)",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No successful predictions yet. Check API health and simulator errors.")

with tab2:
    if "predicted_class" in df.columns:
        class_counts = (
            df.set_index("timestamp")
            .groupby([pd.Grouper(freq="min"), "predicted_class"])
            .size()
            .reset_index(name="count")
        )
        fig = px.bar(
            class_counts,
            x="timestamp",
            y="count",
            color="predicted_class",
            color_discrete_map={
                "healthy": "#2ca02c",
                "anomaly": "#ff7f0e",
                "imminent_failure": "#d62728",
            },
            title="Répartition healthy / anomaly / imminent_failure par minute",
        )
        st.plotly_chart(fig, use_container_width=True)

st.subheader(
    f"Drift par capteur (fenêtre des {window_size} dernières requêtes vs référence train)"
)

drift_table = pd.DataFrame(
    [
        {
            "Capteur": r.feature,
            "PSI": round(r.psi, 3),
            "Niveau PSI": r.psi_level,
            "KS p-value": round(r.ks_pvalue, 4),
            "Drift détecté": "🚨 oui" if r.is_drifted else "—",
        }
        for r in sorted(drift_results, key=lambda r: r.psi, reverse=True)
    ]
)


def highlight_level(row):
    color = {
        "alert": "background-color: #ffcccc",
        "warning": "background-color: #fff3cd",
        "ok": "",
    }
    return [color.get(row["Niveau PSI"], "")] * len(row)


st.dataframe(drift_table.style.apply(highlight_level, axis=1), use_container_width=True)

st.subheader("Comparaison de distribution — référence (train) vs fenêtre courante")
selected_feature = st.selectbox("Capteur à inspecter", FEATURES)

ref_values = reference["features"][selected_feature]["values"]
cur_values = (
    current_df[selected_feature].dropna().tolist()
    if selected_feature in current_df.columns
    else []
)

fig = go.Figure()
fig.add_trace(
    go.Histogram(
        x=ref_values, name="Référence (train)", opacity=0.6, histnorm="probability density"
    )
)
fig.add_trace(
    go.Histogram(
        x=cur_values, name="Fenêtre courante", opacity=0.6, histnorm="probability density"
    )
)
fig.update_layout(barmode="overlay", title=f"Distribution — {selected_feature}")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Historique du drift agrégé (PSI moyen par fenêtre glissante de 20 requêtes)")

if len(df) >= 20:
    rolling_psi = []
    step = 10
    for end in range(20, len(df) + 1, step):
        window_df = df.iloc[max(0, end - 20) : end].rename(
            columns={f"reading_{f}": f for f in FEATURES}
        )
        psis = []
        for feature in FEATURES:
            if feature in window_df.columns and window_df[feature].notna().any():
                psi = compare_all_features(reference_df, window_df, [feature])
                if psi:
                    psis.append(psi[0].psi)
        rolling_psi.append({"n_requetes": end, "psi_moyen": np.mean(psis) if psis else 0})

    rolling_df = pd.DataFrame(rolling_psi)
    fig = px.line(
        rolling_df,
        x="n_requetes",
        y="psi_moyen",
        title="PSI moyen (tous capteurs) au fil du temps",
    )
    fig.add_hline(
        y=PSI_THRESHOLD_ALERT, line_dash="dash", line_color="red", annotation_text="seuil d'alerte"
    )
    fig.add_hline(
        y=PSI_THRESHOLD_WARNING,
        line_dash="dot",
        line_color="orange",
        annotation_text="seuil de surveillance",
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Pas assez de requêtes encore pour tracer l'historique de drift (minimum 20).")
