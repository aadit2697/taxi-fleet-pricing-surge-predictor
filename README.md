# NYC Taxi Surge Pricing — Demand Forecasting & Dynamic Pricing

A data-driven strategy for maximising an NYC taxi fleet's daily revenue with dynamic
(surge) pricing, built on the NYC TLC Yellow Taxi trip records for January 2024. Regular
yellow-cab fares are regulated by the TLC — nothing in this data was ever actually
surge-priced — so the project treats the fleet as a **hypothetical dynamically-priced
operator** layered on top of real historical demand.

The pipeline: (1) forecast next-24h trip demand for two key locations, (2) model how
riders would respond if a price multiplier were applied on top of the base fare, and
(3) recommend, hour by hour and zone by zone, the multiplier that maximises expected
revenue without pushing away enough riders to lose money.

## Contents

| Path | What it is |
|---|---|
| `notebooks/taxi_surge_pricing.ipynb` | The full case study: EDA, demand forecasting, pricing model, sensitivity analysis, A/B test design. Source of truth. |
| `pricing_lib.py` | The elasticity/revenue math (calibration, `P(m)`, `expected_revenue`), shared by the notebook and the dashboard so they never drift apart. |
| `streamlit_app.py` | An executive dashboard: live "what-if" surge recommendations off the notebook's fitted forecast, with sliders to stress-test the elasticity assumption. |
| `data/` | Trip data and the notebook's exported pricing inputs (see [Data](#data)). |
| `requirements.txt` | Pinned dependencies. |

## Data

NYC TLC Yellow Taxi Trip Records, January 2024 — single month, no external data.
Two key pickup zones are analysed in depth: **JFK Airport** and **Times Sq/Theatre
District**.

- `data/taxi_zone_lookup.csv` — TLC zone ID → name lookup (tracked).
- `data/pricing_input.parquet` — the notebook's fitted forecast + elasticity table,
  exported so `streamlit_app.py` can reprice live without refitting the demand model
  (tracked).
- `data/yellow_tripdata_2024-01.parquet` — the full raw trip file (~3M rows, ~50MB).
  **Not tracked** (see `.gitignore`) — re-download it from the
  [NYC TLC trip record data page](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
  and place it at that path to re-run the notebook from Sec 3 onward.

## Requirement coverage

| # | Requirement | Section |
|---|---|---|
| 1a | Core demand patterns: daily/weekly trends, busiest locations | Sec 5 — EDA, demand patterns |
| 1b | Relationship between trip distance, time of day, and fare | Sec 5 — EDA, fare structure |
| 2a | Hourly aggregation for key `PULocationID`s (JFK, Times Sq) | Sec 6 — build hourly panel |
| 2b | 24h demand forecast model, justified | Sec 6 — model, backtest, forecast, justification |
| 3a | Model surge-multiplier effect on revenue; price elasticity | Sec 7 — elasticity model |
| 3b | State assumptions clearly | Sec 7 — assumptions + sensitivity |
| 3c | Recommend optimal multiplier by forecast/time/location | Sec 7 — optimiser |
| 3d | Use the given Expected Revenue formula | Sec 7 — revenue engine |
| 4a | A/B test design to validate on a fleet subset | Sec 8 — experiment design |
| 4b | KPIs beyond total revenue | Sec 8 — KPIs |

## Key findings

### Data quality
- 2,964,624 raw trip rows; **95.84% retained** (2,841,381 rows) after removing stray
  timestamps, non-positive fares, implausible distances, and unknown zones.
- **VendorID 1 double-counts the congestion surcharge** — its fare components overshoot
  `total_amount` by a median of ~$2.50, while VendorID 2 reconciles almost exactly
  (96% of rows within 2¢). Revenue is therefore modelled on `fare_amount`, not
  `total_amount`.
- JFK trips (`RatecodeID=2`) are a flat fare (97.9% cluster in $50–75, no distance
  relationship) and are handled separately from metered trips throughout.

### Demand patterns
- Two overlapping cycles — a within-day commute/nightlife curve and a weekday-vs-weekend
  shift — visible in both zones; JFK and Times Sq both rank in the citywide top-10
  busiest pickup zones by volume.
- Fares track `max(distance, time)`, not distance alone — fare-per-mile rises measurably
  during congested hours, confirming a real time-based meter component on top of the
  per-mile rate.
- **Recorded trips are fulfilled demand, not true demand.** Both key zones show a
  narrowing spread at the top of their hourly-count distribution (90th percentile close
  to the 75th) — a signature of a supply ceiling at peak hours. This means the demand
  forecast, and the revenue lift estimated from it, are likely **conservative** exactly
  at the busiest hours, where surge matters most.
- No demand-responsive pricing exists in this data to begin with: trip volume moved
  **+17.3%** from the quietest to busiest week of January, while the underlying
  uncongested per-mile rate moved only **+0.3%** — essentially flat. This is why price
  elasticity is *modelled*, not estimated from the data.

### Demand forecasting
- Model: gradient-boosted trees (`HistGradientBoostingRegressor`) on lag (1, 2, 3, 24,
  168h) and calendar features, log1p-transformed target, trained per zone-hour.
- Validated with a rolling-origin (expanding-window) backtest against a seasonal-naive
  baseline (same hour, one week ago) — never random k-fold, since that would leak
  future information into the past.
- **Beats the seasonal-naive baseline at every forecast horizon out to 24h.** Overall
  backtest WAPE: **16.4%** (model) vs. **19.3%** (naive) — a **14.7%** overall error
  reduction, with per-horizon improvement ranging roughly 3–40%.
- Permutation importance confirms the model actually relies on the signals it's meant
  to: `lag_168` (same hour last week) and `lag_1` (an hour ago) dominate, together
  accounting for more than half of total feature importance; calendar flags
  (`is_weekend`, `is_holiday`) score near zero not because they don't matter, but
  because `lag_168` already encodes that information directly.
- Alternative considered: SARIMAX with Fourier terms — rejected because a single
  seasonal period can't represent both the 24h and 168h cycles at once without the same
  added complexity the tree model handles natively via lag features.

### Pricing / elasticity model
- A logit (discrete-choice) demand model: `P(m) = 1 / (1 + exp(-(α − βF·m)))`, calibrated
  from two interpretable anchors — elasticity at 1.0x (`eps_base ≈ 0.6`, anchored to
  published ride-hailing estimates such as UberX ≈ −0.5 to −0.6) and base win-share
  (`p1 ≈ 0.5`). Chosen over a constant-elasticity curve (no interior revenue optimum) or
  a linear curve (arbitrary shape).
- **Sanity check passed:** the unconstrained revenue-maximising multiplier is **1.37x**,
  where elasticity is exactly **−1.00** — confirming the model is wired correctly
  (revenue is maximised where |elasticity| = 1, by definition).
- Elasticity is **segmented** by observed rider mix per zone-hour: airport pickups and
  high-tippers are treated as less price-sensitive (×0.60, ×0.85), short trips (<1 mi)
  and late-night weekend trips as more price-sensitive (×1.40, ×1.20) — weighted
  averages, renormalised so the fleet-wide mean still equals `eps_base`.
- **Demand-pressure adjustment:** elasticity is further scaled by how busy the forecast
  hour is vs. that zone's own historical median for that hour (`demand_pressure`,
  observed range **0.50x–1.63x** typical-for-hour across the two zones) — busier hours
  are treated as less price-sensitive, since riders have fewer competing options.

### Recommended pricing & revenue impact
- Over the 24h forecast window across both zones: expected revenue at flat 1.0x pricing
  is **$270,131**; at the recommended per-zone-hour multipliers, **$291,358** — a
  **+7.9% total revenue lift**, from a 5-point multiplier grid (`1.00x`–`2.00x`) with
  light 3-hour smoothing so riders don't see abrupt swings.

### Sensitivity analysis
- Swept `eps_base` over [0.30, 1.20] (half to double the anchor) and `p1` over
  [0.35, 0.65]. The exact-label match rate across that grid is low (0.0% of zone-hours
  keep the identical multiplier at every combination) — expected, given a coarse 5-point
  grid and a deliberately wide stress range.
- The more decision-relevant number is **revenue regret**: pricing on the base-case
  recommendation instead of the true elasticity costs on average **8.3%** of achievable
  revenue (worst single zone-hour: **20.0%**) — the revenue curve is fairly flat near its
  optimum, so a "wrong" elasticity guess is a bounded, quantified mistake rather than an
  unknown one.

### A/B test design
- Rider-level randomisation is invalid here (a classic SUTVA/interference violation —
  shared driver pools mean one rider's price affects nearby riders' experience). Design:
  a **switchback experiment**, randomising each zone × 2-hour time-block independently
  to surge-on/off, with carryover washout periods and standard errors clustered at the
  switchback-unit level.
- Sample size was grounded in this dataset's actual block-to-block revenue volatility
  (coefficient of variation **≈0.61** at both zones — nearly double an initial
  illustrative guess of 0.35) rather than assumed. Detecting a 5% revenue lift at 80%
  power needs roughly **2,359 blocks/arm** (JFK) — about **197 days** if that zone alone
  gates the pilot. Levers to shorten this: accept a larger MDE (10% cuts required
  blocks ~4x), use shorter blocks, or run more pilot zones in parallel.

### KPIs beyond revenue
- **Guardrails** (pause the rollout if breached): trip cancellation rate, customer
  wait time/ETA, fulfilment rate.
- **Health metrics**: driver utilisation, driver earnings per online hour, repeat-rider
  rate, complaint rate, share of trips exposed to surge, and live-tracked forecast error
  (a forecast drifting outside its backtest range should flag the pricing numbers as
  unreliable).

### Stated limitations
- **Censored demand** — every count is *fulfilled* demand; true demand and achievable
  revenue lift are probably understated at the busiest, most surge-relevant hours.
- **Elasticity is assumed, not estimated** — regulated fares mean there is no real price
  variation in this data to measure from; the model is calibrated to external anchors
  and stress-tested, not fit to observed price response.
- **Zones priced independently** — no modelling of drivers relocating between zones in
  response to price, even though a same-dataset check shows pickup/dropoff imbalance
  already happening today.
- **Single month of data** — January 2024 alone can't separate a stable weekly pattern
  from one-off events (two January holidays fall inside it); a production version should
  validate against additional months.

## Running it

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Full case study (re-run all cells to regenerate data/pricing_input.parquet)
jupyter lab notebooks/taxi_surge_pricing.ipynb

# Executive dashboard (reads data/pricing_input.parquet)
streamlit run streamlit_app.py
```
