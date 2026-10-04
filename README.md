# Équinoxe Rent Intelligence

> Explainable, leakage-safe forecasting of 2026 effective rent growth for Collection Équinoxe.


Équinoxe Rent Intelligence is a Python/Jupyter forecasting system developed for the **JADCO / Collection Équinoxe – Clés en main challenge at CodeML 2026**.

The objective is to estimate the **2026 rent increase** using historical lease information while respecting the operational realities of the portfolio: lease transitions, concessions, unit-level history, property structure, limited point-in-time information, and uncertainty.

Rather than optimizing a single black-box model, the project builds an **explainable expert-based forecasting architecture** in which different models specialize in stability, structural information, and recent regime detection.

**Submission Note:** The final project was submitted on Devpost before the deadline. Due to a ZIP upload issue, the archive could not be attached, so the GitHub repository link was provided instead. Commit a0e10db contains the final competition implementation; this README was added afterward solely for documentation and navigation, with no changes to the submitted models, code, results, or 2026 forecast.
---

## Final 2026 Forecast

### P1 Forecast

**2.445%**

| Scenario | Forecast |
|---|---:|
| Downside | **0.781%** |
| Base | **2.445%** |
| Upside | **4.109%** |

Additional 2026 diagnostics:

- Forecast cohort: **931 units**
- Québec units: **810**
- Ontario units: **121**
- Expected lease-transition occurrence: **94.922%**
- Conditional effective-rent growth: **2.566%**
- Regime expert activation: **Inactive due to insufficient point-in-time support**

The scenario range is designed as a decision-support sensitivity range and is **not presented as a statistically calibrated confidence interval**.

---

# What We Built

The system follows the complete forecasting pipeline:

```text
Raw historical data
        ↓
Data audit & integrity checks
        ↓
Leakage-safe unit reconstruction
        ↓
Same-unit transition engine
        ↓
Occurrence × Conditional Growth decomposition
        ↓
Expert A — Stability
Expert B — Structure
Expert C — Regime
        ↓
Point-in-time trust engine
        ↓
Contextual expert orchestration
        ↓
Backtesting 2023–2025
        ↓
2026 forecast
        ↓
LOW / BASE / HIGH decision scenarios
```

The main design principle is simple:

> A more complex model should only influence the forecast when the information available at prediction time provides enough evidence to trust it.

---

# 1. Data Foundation

The first stage focused on building a reliable forecasting foundation rather than immediately fitting models.

### Unit identity

Units are reconstructed using the composite identifier:

```text
sPropCode + sUnitCode
```

This avoids collisions that occur when unit codes are interpreted without their property context.

### Effective rent

The forecasting target prioritizes **effective rent**, allowing concessions and lease economics to be represented more realistically than with contractual rent alone.

### Same-unit analysis

The system focuses on **same-unit transitions** instead of comparing aggregate portfolio medians across years.

This is important because portfolio composition can change over time and create apparent rent growth that is not caused by actual same-unit rent changes.

---

# 2. Leakage-Safe Forecasting

All historical evaluations are performed under a **point-in-time information constraint**.

For each forecast year, the model only uses information that would have been available at the corresponding historical cutoff.

Forecast origins:

| Forecast Year | Information Cutoff |
|---|---|
| 2023 | 2022-12-31 |
| 2024 | 2023-12-31 |
| 2025 | 2024-12-31 |
| 2026 | 2025-12-31 |

This prevents future information from leaking into historical forecasts.

---

# 3. Forecast Decomposition

Instead of directly predicting one rent-growth number, the system decomposes the problem into two components.

For unit `i`:

```text
Expected contribution_i = p_i × g_i
```

where:

- `p_i` = probability that the relevant lease transition occurs
- `g_i` = expected conditional effective-rent growth if that transition occurs

The portfolio P1 forecast is then derived from the distribution of unit-level expected contributions.

This decomposition makes the forecast easier to interpret and allows occurrence risk and rent-growth risk to be modeled independently.

---

# 4. Expert A — Stability

Expert A represents the **long-run robust baseline**.

Its purpose is not to react aggressively to recent observations. Instead, it provides a stable forecasting anchor using historical information available at the forecast cutoff.

The project tested more complex alternatives against these simple baselines rather than assuming machine learning would automatically improve forecasting accuracy.

Expert A ultimately remains the dominant source of trust when specialist evidence is weak.

---

# 5. Expert B — Structural Context

Expert B captures **hierarchical and portfolio structure**.

It allows the system to use more localized information when enough observations exist while shrinking toward broader historical information when support is limited.

Conceptually:

```text
local evidence
      ↓
hierarchical support
      ↓
shrinkage toward global evidence
```

The amount of influence given to Expert B depends on its statistical support and how much genuinely different information it contributes relative to the stable baseline.

This prevents small groups from receiving excessive influence simply because they produce a different prediction.

---

# 6. Expert C — Regime Detection

Expert C was designed to capture **recent momentum or regime changes**.

However, the architecture explicitly prevents recent information from being used merely because it exists.

Before Expert C can influence the forecast, it must satisfy qualification requirements including:

- minimum sample support,
- sufficient coverage,
- point-in-time availability,
- internal coherence,
- meaningful divergence from the baseline.

For the final 2026 forecast, the available recent information did **not** meet the required support thresholds.

Therefore:

```text
Expert C weight = 0
```

This is intentional.

The system prefers to report **insufficient evidence** rather than manufacture a recent-trend signal from a small sample.

---

# 7. Contextual Expert Orchestration

The final system is not a winner-take-all model tournament.

Experts are combined through a **trust engine**.

Occurrence and conditional growth receive separate expert weights:

```text
p_i = Σ w(E,p,i) × p(E,i)

g_i = Σ w(E,g,i) × g(E,i)
```

The weights are determined from information available at the prediction cutoff.

### Expert A

Always provides the safe fallback:

```text
t_A = 1
```

### Expert B

Trust depends on hierarchical support and useful divergence from the baseline.

### Expert C

Trust requires explicit qualification based on sample size, coverage, coherence and divergence.

If specialists do not provide sufficiently supported information, the system naturally falls back toward Expert A.

---

# 8. Models Were Allowed to Fail

An important part of the project was rejecting approaches that did not provide evidence of improvement.

For example, an additional building-level growth shrinkage model was evaluated with a fixed ex-ante shrinkage parameter.

It failed to improve the historical forecasts and was therefore rejected.

Similarly, simple fixed averaging of Experts A and B was investigated. Their historical errors were strongly related, meaning averaging them did not create enough new information to justify treating the combination as a new forecasting solution.

These experiments remain part of the project because unsuccessful models provide useful evidence about the structure of the forecasting problem.

---

# 9. Historical Backtesting

The complete forecasting process was replayed for:

- **2023**
- **2024**
- **2025**

using only information available at each historical forecast origin.

Final contextual orchestration performance:

| Year | Forecast P1 | Realized P1 | Signed Error |
|---|---:|---:|---:|
| 2023 | 2.872% | 2.036% | +0.836 pp |
| 2024 | 3.272% | 1.608% | +1.664 pp |
| 2025 | 2.973% | 3.301% | -0.328 pp |

Aggregate diagnostics:

- MAE: **0.943 percentage points**
- RMSE: **1.092 percentage points**
- Historical bias: **+0.724 percentage points**
- Maximum historical absolute error: **1.664 percentage points**

The backtests are also used diagnostically to understand **why** the model misses, rather than only reporting an aggregate score.

---

# 10. 2026 Expert Trust

For the final 2026 forecast:

### Occurrence

- Expert A average trust weight: **92.690%**
- Expert B average trust weight: **7.310%**
- Expert C: **0%**

### Conditional growth

- Expert A average trust weight: **98.436%**
- Expert B average trust weight: **1.564%**
- Expert C: **0%**

The resulting estimates are:

```text
Expected occurrence = 94.921771%
Conditional growth = 2.566148%
Final P1 = 2.445012%
```

---

# 11. External Signals

External public information was audited as potential forecasting context.

External information is only incorporated when it can be aligned with the historical point-in-time framework and supported consistently enough to avoid introducing leakage or arbitrary hindsight adjustments.

The project therefore distinguishes between:

- information that is economically interesting,
- information that is historically available,
- information that is actually qualified for forecasting.

External variables are **not forced into the model simply because they are available**.

---

# 12. Explainability and Reproducibility

The repository contains the complete modeling workflow:

```text
data/external/     Public external signals
docs/              Methodology and technical documentation
notebooks/         Analysis, experiments and final submission notebook
outputs/           Aggregated forecasting outputs
src/               Forecasting implementation
tests/             Automated validation
```

The final production entry points include:

```text
src/models/final_submission.py
notebooks/14_final_submission.ipynb
outputs/reports/final_2026_summary.json
docs/final_submission.md
```

The final implementation exposes the required forecasting and backtesting functionality, including:

```python
estimate_2026()
backtest()
```

---

# 13. Validation

Before finalization, the project was tested at multiple levels:

- data integrity checks,
- leakage checks,
- forecasting component tests,
- orchestration tests,
- historical backtest reproduction,
- final forecast validation,
- notebook execution.

Final test suite:

```text
212 tests passed
```

The final submission notebook was also executed from top to bottom without errors.

---

# 14. Technology

Built with:

- Python
- Jupyter Notebook
- pandas
- NumPy
- scikit-learn
- pytest
- Git
- GitHub

---

# Data Confidentiality

The confidential JADCO / Collection Équinoxe CRM dataset is **not included in this public repository**.

Only code, documentation, aggregated outputs, and permitted external/public data are versioned here.

The forecasting pipeline expects the challenge data to be supplied separately in the appropriate local environment.

---

# Submission Integrity Note

The **final competition implementation was completed and submitted on Devpost before the competition deadline**.

The final implementation is represented by Git commit:

```text
a0e10db
feat: finalize contextual orchestration and 2026 forecast
```

This commit contains the final forecasting implementation, notebooks, tests, documentation and 2026 results.

A ZIP archive of the repository had also been prepared for submission, but due to an upload issue it was not attached to the Devpost submission. The GitHub repository URL was provided instead.

After the deadline, this README was added solely to make the already-submitted repository easier for evaluators to navigate and understand.

**No forecasting code, model parameters, model selection, backtest results, input data, or 2026 forecast results were modified as part of the post-deadline README update.**

The Git history provides the versioned project chronology, with commit `a0e10db` identifying the final competition implementation prior to this documentation-only update.

---

# Final Result

> **Équinoxe Rent Intelligence forecasts a 2026 P1 effective-rent growth contribution of 2.445%, with a decision-support range of 0.781% to 4.109%.**

The central contribution of the project is not only the final number.

It is a forecasting framework designed to answer three questions:

1. **What does the historical evidence suggest?**
2. **How much should we trust each source of information?**
3. **When is the available evidence too weak to justify additional model complexity?**

That philosophy drives the complete system: **stability first, specialization when supported, and no artificial confidence when the data does not justify it.**