from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "car_data.csv"
MODEL_PATH = PROJECT_ROOT / "model" / "car_price_model.pkl"


# ============================================================
# 1. Load data
# ============================================================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 2. Basic cleaning
# ============================================================

required_columns = [
    "name",
    "year",
    "selling_price",
    "km_driven",
    "fuel",
    "seller_type",
    "transmission",
    "owner",
]

df = df[required_columns].copy()

# Remove rows containing missing values
df = df.dropna()

# Remove impossible / invalid values
df = df[df["selling_price"] > 0]
df = df[df["km_driven"] >= 0]
df = df[df["year"] >= 1990]


print(f"\nDataset after cleaning: {df.shape}")


# ============================================================
# 3. Features and target
# ============================================================

X = df[
    [
        "name",
        "year",
        "km_driven",
        "fuel",
        "seller_type",
        "transmission",
        "owner",
    ]
]

y = df["selling_price"]


# ============================================================
# 4. Train/Test split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)


print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")


# ============================================================
# 5. Feature types
# ============================================================

categorical_features = [
    "name",
    "fuel",
    "seller_type",
    "transmission",
    "owner",
]

numerical_features = [
    "year",
    "km_driven",
]


# ============================================================
# 6. Preprocessing
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features,
        ),
        (
            "numerical",
            "passthrough",
            numerical_features,
        ),
    ]
)


# ============================================================
# 7. ML model
# ============================================================

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
)


# ============================================================
# 8. Complete ML pipeline
# ============================================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ]
)


# ============================================================
# 9. Train
# ============================================================

print("\nTraining model...")

pipeline.fit(X_train, y_train)


# ============================================================
# 10. Evaluation
# ============================================================

y_pred = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n==============================")
print("MODEL EVALUATION")
print("==============================")

print(f"MAE: ₹{mae:,.2f}")
print(f"R²:  {r2:.4f}")


# ============================================================
# 11. Save complete pipeline
# ============================================================

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(
    pipeline,
    MODEL_PATH,
)

print("\n==============================")
print("MODEL SAVED")
print("==============================")

print(MODEL_PATH)
