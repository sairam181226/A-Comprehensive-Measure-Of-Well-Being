# A Comprehensive Measure of Well-Being
### Predicting the Human Development Index (HDI) Tier from Health, Education, and Income Indicators

## 1. Background

The Human Development Index (HDI) is a composite statistic built from three
dimensions of human development:

| Dimension        | Indicator(s)                                              |
|-------------------|-------------------------------------------------------------|
| Health            | Life expectancy at birth                                    |
| Education         | Mean years of schooling (adults 25+) & expected years of schooling (children entering school) |
| Standard of Living | Gross National Income (GNI) per capita, PPP-adjusted        |

Each dimension is converted into a 0–1 sub-index, and the overall HDI is the
**geometric mean** of the three sub-indices — not a simple average — which
means a country cannot compensate for a very weak dimension (e.g. low life
expectancy) purely with strength in another (e.g. high income).

Countries are then classified into four development tiers using the
official UNDP cutoffs:

| Tier       | HDI range       |
|------------|-----------------|
| Very High  | ≥ 0.800         |
| High       | 0.700 – 0.799   |
| Medium     | 0.550 – 0.699   |
| Low        | < 0.550         |

## 2. Project Goal

Build a system that takes the four raw, easy-to-obtain indicators —
life expectancy, mean years of schooling, expected years of schooling, and
GNI per capita — and predicts which of the four development tiers a country
falls into, matching the three scenarios in the brief:

- **Scenario 1** — strong indicators across the board → *Very High*
- **Scenario 2** — mid-range indicators → *Medium*, with gap analysis
- **Scenario 3** — weak indicators → *Low*, flagged for intervention

## 3. Data

Real-time UNDP country data wasn't available in this environment, so a
**synthetic dataset of 220 countries** (`data/countries_dataset.csv`) was
generated to be statistically realistic:

- Countries are drawn from four development "regimes" (very high / high /
  medium / low) with within-regime random variation, so the dataset spans
  the full real-world range of each indicator.
- The **true HDI and tier label** for every row is computed using the exact
  official UNDP formula (see `generate_dataset.py`), so the ground truth is
  methodologically correct, not arbitrary.
- Small measurement noise is added to the raw indicators to keep the
  learning task realistic rather than a pure algebra-inversion exercise.

Resulting class balance: Medium 58, Low 58, Very High 53, High 51 — reasonably
balanced across the four tiers.

## 4. Modeling

Two classifiers were trained on a 75/25 train/test split (`train_model.py`):

| Model                | Test Accuracy |
|-----------------------|---------------|
| Logistic Regression   | **89.1%**     |
| Random Forest         | 87.3%         |

Key preprocessing step: GNI per capita is **log-transformed** before
modeling. This matters because the true HDI formula uses `log(GNI)`, not
raw GNI — without the log transform, a linear model in particular
would badly under-fit the income dimension.

Logistic Regression edged out Random Forest here because the underlying
relationship (geometric mean of monotonic sub-indices) is close to linear
once GNI is log-transformed and features are standardized. Most
misclassifications are between **adjacent** tiers (e.g. High vs. Very High),
which is expected — the sharpest errors happen right at the UNDP cutoff
boundaries, which is a natural limit of any threshold-based tiering system.

See `figures/confusion_matrix.png` and `figures/feature_importance.png`.

## 5. Deliverables

| File                                   | Purpose                                                   |
|------------------------------------------|-------------------------------------------------------------|
| `data/countries_dataset.csv`            | 220-row synthetic country dataset with ground-truth HDI     |
| `generate_dataset.py`                   | Reproducible dataset generator (official UNDP formula)      |
| `train_model.py`                        | Trains & evaluates Logistic Regression / Random Forest      |
| `models/hdi_classifier.joblib`          | Saved best model + label encoder                              |
| `figures/confusion_matrix.png`          | Test-set confusion matrix                                    |
| `figures/feature_importance.png`        | Random Forest feature importances                            |
| `hdi_explorer.html`                     | Interactive calculator/predictor web app (see below)         |
| `report.md`                             | This document                                                 |

## 6. Interactive Explorer

`hdi_explorer.html` is a self-contained, no-install web app that lets a user
(policymaker, researcher, or student) enter the four raw indicators with
sliders and instantly see:

- The computed sub-indices (Health, Education, Income) as a visual breakdown
- The overall HDI score
- The predicted development tier, styled to match the UNDP tier colors
- Contextual guidance matching the three brief scenarios (e.g. flags "areas
  for improvement" when a country lands in Medium or Low)

It implements the same official formula used to label the training data, so
its predictions are exact rather than approximate — this mirrors how a
production version of this tool would likely combine a transparent,
auditable formula (for the score itself) with the ML model (for pattern
discovery, comparison, and forecasting from partial/noisy data).

## 7. Suggested Next Steps

- Replace the synthetic dataset with real UNDP Human Development Report data
- Add a time dimension to show a country's HDI trajectory over years
- Add inequality-adjusted HDI (IHDI), Gender Development Index (GDI), and
  Multidimensional Poverty Index (MPI) as companion measures
- Expose the trained classifier via a small API so `hdi_explorer.html` can
  call the actual ML model in addition to the formula-based calculation
