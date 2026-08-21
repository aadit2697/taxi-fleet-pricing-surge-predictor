"""Executive dashboard for the NYC taxi surge-pricing case study.

Reads the fitted pricing inputs (`data/pricing_input.parquet`, produced by
notebooks/taxi_surge_pricing.py Sec 7.6) and reruns only the cheap part of the
pipeline -- the multiplier optimiser -- live, using the exact same math
(pricing_lib.py) as the notebook. The slow part (the 24h demand forecast) is
precomputed once in the notebook, not refit here.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from pricing_lib import optimise_row, MULTIPLIERS

DATA_PATH = Path(__file__).parent / "data" / "pricing_input.parquet"

BLUE_RAMP = ["#b7d3f6", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"]  # sequential, light -> dark
ZONE_COLORS = {"JFK Airport": "#2a78d6", "Times Sq/Theatre District": "#eb6834"}  # categorical slots 1, 2
GOOD = "#0ca30c"

st.set_page_config(page_title="Surge Pricing — Executive View", layout="wide")


@st.cache_data
def load_pricing_input() -> pd.DataFrame:
    if not DATA_PATH.exists():
        st.error(
            f"Missing {DATA_PATH.name}. Run notebooks/taxi_surge_pricing.py (or open the paired "
            "notebook and run all cells) once to generate it -- this app doesn't refit the demand "
            "forecast itself."
        )
        st.stop()
    return pd.read_parquet(DATA_PATH)


@st.cache_data
def price(df: pd.DataFrame, eps_col: str, eps_base_used: float | None, p1_used: float | None) -> pd.DataFrame:
    """Sweep the multiplier grid for every zone-hour. Cheap (48 rows) -- safe to call on every interaction."""
    out = df.copy()
    results = out.apply(
        lambda r: pd.Series(
            optimise_row(r, eps_col=eps_col, eps_base_used=eps_base_used, p1_used=p1_used),
            index=["rev_1x", "rev_best", "best_m"],
        ),
        axis=1,
    )
    return pd.concat([out, results], axis=1)


df = load_pricing_input()
zones = list(df["zone"].unique())

st.title("Surge Pricing — Executive View")
st.caption(
    "NYC TLC Yellow Taxi, January 2024 · JFK Airport & Times Sq/Theatre District · "
    "next-24h forecast · recommended multiplier per zone-hour"
)

with st.sidebar:
    st.header("Controls")
    demand_responsive = st.toggle(
        "Demand-responsive pricing",
        value=True,
        help="ON: multiplier also reacts to how busy the forecast hour is vs. that zone's own "
        "history. OFF: multiplier depends only on rider mix (airport vs. short-hop vs. late-night), "
        "flat across the whole day within a zone -- today's starting point.",
    )
    st.divider()
    st.subheader("Stress-test the elasticity assumption")
    st.caption(
        "The multiplier depends on an assumed price sensitivity, anchored to a published estimate, "
        "not measured from this data (nothing here was ever actually surge-priced). Move these to "
        "see how much the recommendation -- and the revenue lift -- depends on that guess."
    )
    eps_base = st.slider("Elasticity anchor (eps₀)", 0.30, 1.20, 0.60, 0.05)
    p1 = st.slider("Base win-share (p₁)", 0.35, 0.65, 0.50, 0.01)
    stress_test_active = (abs(eps_base - 0.60) > 1e-9) or (abs(p1 - 0.50) > 1e-9)
    if stress_test_active:
        st.info("Stress-test values applied below (overrides the fitted, zone/hour-specific elasticity).")

eps_col = "eps_effective" if demand_responsive else "eps_seg"
eps_override = eps_base if stress_test_active else None
p1_override = p1 if stress_test_active else None

priced = price(df, eps_col, eps_override, p1_override)
priced_baseline = price(df, "eps_seg", None, None)  # segmented-only, no demand response, no override

rev_1x = priced["rev_1x"].sum()
rev_best = priced["rev_best"].sum()
rev_baseline_best = priced_baseline["rev_best"].sum()
lift_vs_1x = rev_best / rev_1x - 1
uplift_vs_baseline = rev_best - rev_baseline_best
uplift_pct = uplift_vs_baseline / rev_baseline_best

c1, c2, c3 = st.columns(3)
c1.metric("Revenue @ 1.0x (no surge)", f"${rev_1x:,.0f}")
c2.metric("Revenue @ recommended pricing", f"${rev_best:,.0f}", delta=f"{lift_vs_1x:+.1%} vs. 1.0x")
c3.metric(
    "Uplift from demand-responsiveness",
    f"${uplift_vs_baseline:,.0f}",
    delta=f"{uplift_pct:+.1%} vs. segmented-only pricing",
    help="Segmented-only = multiplier varies by zone's rider mix but not by how busy the hour is "
    "(today's starting point). This is the extra revenue from also reacting to demand.",
)

tab_heatmap, tab_zones = st.tabs(["Heatmap", "By zone, hour-by-hour"])

with tab_heatmap:
    pivot_m = priced.pivot(index="hour", columns="zone", values="best_m").reindex(columns=zones).sort_index()
    pivot_fc = priced.pivot(index="hour", columns="zone", values="forecast").reindex(columns=zones).sort_index()
    pivot_dp = priced.pivot(index="hour", columns="zone", values="demand_pressure").reindex(columns=zones).sort_index()
    pivot_eps = priced.pivot(index="hour", columns="zone", values=eps_col).reindex(columns=zones).sort_index()

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot_m.values,
            x=pivot_m.columns,
            y=pivot_m.index,
            zmin=1.0,
            zmax=2.0,
            colorscale=[
                [0.0, BLUE_RAMP[0]], [0.25, BLUE_RAMP[1]], [0.5, BLUE_RAMP[2]],
                [0.75, BLUE_RAMP[3]], [1.0, BLUE_RAMP[4]],
            ],
            text=pivot_m.values,
            texttemplate="%{text:.2f}x",
            textfont={"size": 13},
            customdata=np.dstack([pivot_fc.values, pivot_dp.values, pivot_eps.values]),
            hovertemplate=(
                "<b>%{x}</b>, hour %{y}:00<br>"
                "Recommended multiplier: <b>%{z:.2f}x</b><br>"
                "Forecast trips: %{customdata[0]:.0f}<br>"
                "Demand pressure: %{customdata[1]:.2f}x typical-for-hour<br>"
                "Elasticity used: %{customdata[2]:.2f}"
                "<extra></extra>"
            ),
            colorbar=dict(title="Multiplier", tickvals=MULTIPLIERS),
            xgap=2,
            ygap=2,
        )
    )
    fig.update_yaxes(autorange="reversed", title="Hour of day", dtick=1)
    fig.update_xaxes(title=None, side="top")
    fig.update_layout(height=680, margin=dict(t=40, b=10))
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    st.caption(
        "Unsmoothed optimiser output (the notebook applies a light 3-hour smoothing pass before "
        "recommending a live schedule, so riders don't see 1.0x → 2.0x → 1.0x swings)."
    )

with tab_zones:
    fig2 = px.line(
        priced.sort_values("hour"),
        x="hour",
        y="best_m",
        color="zone",
        line_shape="hv",
        markers=True,
        color_discrete_map=ZONE_COLORS,
        labels={"hour": "Hour of day", "best_m": "Recommended multiplier", "zone": ""},
    )
    fig2.update_yaxes(range=[0.9, 2.1], dtick=0.25)
    fig2.update_xaxes(dtick=2)
    fig2.update_layout(height=480, legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig2, use_container_width=True, theme="streamlit")

with st.expander("How this multiplier is chosen"):
    st.markdown(
        """
Each zone-hour's multiplier is the one (from `1.00x, 1.25x, 1.50x, 1.75x, 2.00x`) that maximises
`Expected Revenue = Forecasted Trips × (1 − Demand Reduction %) × Avg Fare × Multiplier`, where demand
reduction comes from an assumed price-response curve. Two things feed that curve:

1. **Who's riding** — airport pickups, short hops, and late-night weekend trips are weighted by how
   price-sensitive each group typically is, using the actual historical mix in that zone and hour.
2. **How busy it is** *(the demand-responsive toggle)* — forecast demand is compared against that
   zone's own historical normal for that hour. Busier-than-usual hours are treated as less
   price-sensitive (fewer competing options for riders), which pushes the multiplier up; quieter
   hours push it down.
"""
    )

with st.expander("Assumptions & risks"):
    st.markdown(
        """
- **Price sensitivity is an informed assumption, not a measurement** — nothing in this data was ever
  actually surge-priced, so the elasticity anchor is taken from published ride-hailing estimates.
  Use the sidebar stress-test to see how much the recommendation depends on it.
- **Observed trips are treated as true demand** — at the busiest hours, some demand may already be
  turned away for lack of a cab, which would make the forecast (and this recommendation) conservative
  exactly where surge matters most.
- **Average fare per zone-hour is assumed stable** regardless of the multiplier — in reality, surge
  can shift who rides (e.g. more expensed business travel), which this model doesn't capture.
- **Zones are priced independently** — no modelling of drivers relocating between zones in response
  to price, which real deployment would need to watch for.
"""
    )

st.caption(
    "Model: gradient-boosted 24h demand forecast + segmented logit elasticity + demand-pressure "
    "adjustment. Full methodology in notebooks/taxi_surge_pricing.ipynb."
)
