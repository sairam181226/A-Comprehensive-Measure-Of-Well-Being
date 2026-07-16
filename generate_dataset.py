"""
generate_dataset.py
--------------------
Builds a synthetic dataset of ~220 "countries" with the four raw indicators
used in the UNDP Human Development Index, then computes the *true* HDI and
tier label using the official UNDP methodology (2010 revision):

    Life Expectancy Index (LEI)      = (LE - 20) / (85 - 20)
    Mean Years Schooling Index (MYSI) = MYS / 15
    Expected Years Schooling Index    = EYS / 18
    Education Index (EI)             = (MYSI + EYSI) / 2
    Income Index (II)                = (ln(GNIpc) - ln(100)) / (ln(75000) - ln(100))
    HDI                               = (LEI * EI * II) ** (1/3)

Official UNDP tier cutoffs:
    Very High : HDI >= 0.800
    High      : 0.700 <= HDI < 0.800
    Medium    : 0.550 <= HDI < 0.700
    Low       : HDI < 0.550

A small amount of measurement noise is added to the raw indicators (real-world
statistical offices don't measure these perfectly) so the downstream ML model
has a genuine (if easy) learning task rather than reconstructing an exact
formula.
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 220

# --- 1. Sample archetypal development "regimes", then add within-regime spread ---
# Each regime anchors around real-world plausible ranges (not naming real
# countries, since this is a synthetic teaching dataset).
regimes = [
    # name, LE_range, MYS_range, EYS_range, GNI_range(log-ish), weight
    ("very_high", (76, 85), (11.5, 14.0), (15.5, 18.0), (28000, 90000), 0.22),
    ("high",      (70, 77), (8.5, 11.5),  (13.0, 16.0), (9000, 28000),  0.28),
    ("medium",    (60, 71), (5.0, 8.5),   (9.5, 13.5),  (2000, 10000),  0.30),
    ("low",       (48, 61), (1.5, 5.5),   (5.0, 10.0),  (400, 2500),    0.20),
]

rows = []
for i in range(N):
    idx = RNG.choice(len(regimes), p=[0.22, 0.28, 0.30, 0.20])
    name, le_r, mys_r, eys_r, gni_r, _ = regimes[idx]

    life_expectancy = RNG.uniform(*le_r) + RNG.normal(0, 1.2)
    mean_years_schooling = RNG.uniform(*mys_r) + RNG.normal(0, 0.4)
    expected_years_schooling = RNG.uniform(*eys_r) + RNG.normal(0, 0.5)
    # income sampled log-uniform within regime range, plus noise
    gni_per_capita = np.exp(RNG.uniform(np.log(gni_r[0]), np.log(gni_r[1])))
    gni_per_capita *= RNG.normal(1.0, 0.05)

    # clip to sane real-world bounds
    life_expectancy = float(np.clip(life_expectancy, 40, 86))
    mean_years_schooling = float(np.clip(mean_years_schooling, 0, 15))
    expected_years_schooling = float(np.clip(expected_years_schooling, mean_years_schooling, 20.5))
    gni_per_capita = float(np.clip(gni_per_capita, 300, 120000))

    rows.append(dict(
        country_id=f"C{i+1:03d}",
        life_expectancy=round(life_expectancy, 1),
        mean_years_schooling=round(mean_years_schooling, 2),
        expected_years_schooling=round(expected_years_schooling, 2),
        gni_per_capita=round(gni_per_capita, 0),
    ))

df = pd.DataFrame(rows)

# --- 2. Compute official sub-indices and HDI ---
df["life_expectancy_index"] = (df.life_expectancy - 20) / (85 - 20)
df["mys_index"] = df.mean_years_schooling / 15
df["eys_index"] = df.expected_years_schooling / 18
df["education_index"] = (df.mys_index + df.eys_index) / 2
df["income_index"] = (np.log(df.gni_per_capita) - np.log(100)) / (np.log(75000) - np.log(100))

for col in ["life_expectancy_index", "education_index", "income_index"]:
    df[col] = df[col].clip(0, 1)

df["hdi"] = (df.life_expectancy_index * df.education_index * df.income_index) ** (1 / 3)
df["hdi"] = df["hdi"].round(4)

def tier(hdi):
    if hdi >= 0.800:
        return "Very High"
    elif hdi >= 0.700:
        return "High"
    elif hdi >= 0.550:
        return "Medium"
    else:
        return "Low"

df["hdi_tier"] = df.hdi.apply(tier)

# Keep only the columns a real dataset would ship with (raw indicators + label)
final_cols = [
    "country_id", "life_expectancy", "mean_years_schooling",
    "expected_years_schooling", "gni_per_capita", "hdi", "hdi_tier"
]
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

df_final = df[final_cols].sort_values("hdi", ascending=False).reset_index(drop=True)
output_path = os.path.join(DATA_DIR, "countries_dataset.csv")
df_final.to_csv(output_path, index=False)

print(df_final["hdi_tier"].value_counts())
print(df_final.head(10))
print(f"\nSaved {len(df_final)} rows to data/countries_dataset.csv")
