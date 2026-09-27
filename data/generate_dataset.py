"""
generate_dataset.py
--------------------
Generates a domain-realistic synthetic dataset for the "Irrigation Requirement
Classification" capstone project.

WHY SYNTHETIC DATA?
This project environment has no live internet access at code-execution time,
so a direct `kaggle datasets download` could not be run here. Instead this
script generates a dataset whose schema, feature ranges and feature-target
relationships are modeled on real, publicly documented smart-irrigation
sensor datasets on Kaggle / Mendeley Data (soil moisture from capacitive
soil-moisture sensors, temperature & humidity from DHT11 sensors, crop
growth stage, and recent rainfall) — see README.md -> "Dataset" section for
citations and for exact commands to swap in a real Kaggle CSV.

The generation logic encodes real agronomic domain knowledge (crop
coefficient / Kc by growth stage, soil moisture depletion thresholds,
evapotranspiration driven by temperature & humidity, rainfall reducing
irrigation need) plus realistic sensor noise, missing values, a few
duplicate rows and a handful of outliers -- so that the EDA and preprocessing
stages of this project have real issues to detect and handle.
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_SAMPLES = 2400

rng = np.random.default_rng(RANDOM_SEED)

CROP_STAGES = ["seedling", "vegetative", "flowering", "maturity"]
# Approximate crop coefficient (Kc) per stage -> higher Kc = more water demand
STAGE_KC = {"seedling": 0.4, "vegetative": 0.85, "flowering": 1.15, "maturity": 0.6}
SOIL_TYPES = ["sandy", "loamy", "clayey"]
# Field capacity (approx. max plant-available moisture %) per soil type
SOIL_FIELD_CAPACITY = {"sandy": 28.0, "loamy": 38.0, "clayey": 48.0}

CROP_TYPES = ["wheat", "maize", "rice", "sugarcane", "cotton", "groundnut"]


def generate_raw_dataframe(n=N_SAMPLES, seed=RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    crop_stage = rng.choice(CROP_STAGES, size=n, p=[0.22, 0.33, 0.25, 0.20])
    soil_type = rng.choice(SOIL_TYPES, size=n, p=[0.35, 0.40, 0.25])
    crop_type = rng.choice(CROP_TYPES, size=n)

    # Ambient conditions (seasonal + daily noise)
    day_of_year = rng.integers(1, 366, size=n)
    seasonal_temp = 8 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    temperature_c = 26 + seasonal_temp + rng.normal(0, 3.2, size=n)
    temperature_c = np.clip(temperature_c, 8, 46)

    humidity_pct = 65 - 0.9 * (temperature_c - 26) + rng.normal(0, 8, size=n)
    humidity_pct = np.clip(humidity_pct, 12, 98)

    # Rainfall over the last 7 days (mm) -- right-skewed, many dry spells
    rainfall_mm_7d = rng.gamma(shape=1.1, scale=9.0, size=n)
    rainfall_mm_7d = np.clip(rainfall_mm_7d, 0, 160)

    days_since_last_irrigation = rng.integers(0, 15, size=n)

    field_capacity = np.array([SOIL_FIELD_CAPACITY[s] for s in soil_type])
    kc = np.array([STAGE_KC[s] for s in crop_stage])

    # Reference evapotranspiration proxy: rises with temperature, falls with humidity
    et0 = 0.15 * (temperature_c - 5) - 0.03 * humidity_pct + 2.0
    et0 = np.clip(et0, 0.5, 9)
    crop_water_use = et0 * kc  # mm/day-equivalent depletion rate

    # Soil moisture: starts near field capacity after last irrigation/rain, then
    # depletes with crop water use and days since last watering, replenished by rainfall
    depletion = crop_water_use * (days_since_last_irrigation + 1) * 0.55
    rain_boost = np.minimum(rainfall_mm_7d * 0.35, 0.6 * field_capacity)
    soil_moisture_pct = field_capacity - depletion + rain_boost
    soil_moisture_pct += rng.normal(0, 2.5, size=n)  # sensor noise
    soil_moisture_pct = np.clip(soil_moisture_pct, 2, 60)

    # Wilting point ~ 35% of field capacity (very rough agronomic rule of thumb)
    wilting_point = 0.35 * field_capacity
    # Refill point (trigger threshold) ~ 65% of field capacity
    refill_point = 0.65 * field_capacity

    # --- Target: irrigation_required -------------------------------------
    # Deterministic agronomic signal + logistic noise so classes are not
    # perfectly separable (real sensor/labeling noise).
    signal = (refill_point - soil_moisture_pct) / (refill_point - wilting_point + 1e-6)
    signal += 0.15 * (kc - 0.7)                     # thirstier crop stages push toward irrigation
    signal -= 0.01 * rainfall_mm_7d                 # recent rain reduces need
    signal += 0.01 * (temperature_c - 26)           # heat increases need
    logits = 3.2 * signal + rng.normal(0, 0.55, size=n)  # label noise
    prob_irrigate = 1 / (1 + np.exp(-logits))
    irrigation_required = (prob_irrigate >= 0.5).astype(int)

    df = pd.DataFrame(
        {
            "soil_moisture_pct": soil_moisture_pct.round(2),
            "temperature_c": temperature_c.round(2),
            "humidity_pct": humidity_pct.round(2),
            "rainfall_mm_last_7days": rainfall_mm_7d.round(2),
            "crop_stage": crop_stage,
            "crop_type": crop_type,
            "soil_type": soil_type,
            "days_since_last_irrigation": days_since_last_irrigation,
            "irrigation_required": irrigation_required,
        }
    )
    return df


def inject_data_quality_issues(df: pd.DataFrame, seed=RANDOM_SEED) -> pd.DataFrame:
    """Adds missing values, duplicate rows and a few outliers so the EDA
    section of the project has genuine data-quality problems to find."""
    rng = np.random.default_rng(seed + 1)
    df = df.copy()

    # Missing values (~3% each in a few columns) -- simulate sensor dropouts
    for col, frac in [
        ("humidity_pct", 0.03),
        ("soil_moisture_pct", 0.02),
        ("rainfall_mm_last_7days", 0.025),
        ("crop_type", 0.01),
    ]:
        idx = rng.choice(df.index, size=int(len(df) * frac), replace=False)
        df.loc[idx, col] = np.nan

    # Outliers: a few faulty sensor readings
    out_idx = rng.choice(df.index, size=12, replace=False)
    df.loc[out_idx[:6], "temperature_c"] = rng.uniform(55, 70, size=6)      # sensor spike
    df.loc[out_idx[6:], "soil_moisture_pct"] = rng.uniform(-5, 0, size=6)   # faulty low reading

    # Duplicate rows (~1%)
    dup_rows = df.sample(n=int(len(df) * 0.01), random_state=seed).copy()
    df = pd.concat([df, dup_rows], ignore_index=True)

    # Shuffle
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df.insert(0, "record_id", range(1, len(df) + 1))
    return df


if __name__ == "__main__":
    raw = generate_raw_dataframe()
    final = inject_data_quality_issues(raw)
    out_path = "irrigation_dataset.csv"
    final.to_csv(out_path, index=False)
    print(f"Wrote {len(final)} rows to {out_path}")
    print(final["irrigation_required"].value_counts(normalize=True).round(3))
