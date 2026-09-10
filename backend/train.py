from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import TransformedTargetRegressor


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "car_data.csv"
MODEL_PATH = PROJECT_ROOT / "model" / "car_price_model.pkl"


# ============================================================
# CONFIG
# ============================================================

CURRENT_YEAR = 2026
RANDOM_STATE = 42


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def add_features(df):
    """
    Add useful derived features while keeping the original
    API input fields unchanged.
    """

    df = df.copy()

    # Age of the car
    df["car_age"] = CURRENT_YEAR - df["year"]

    # Extract brand from car name
    df["brand"] = (
        df["name"]
        .astype(str)
        .str.strip()
        .str.split()
        .str[0]
    )

    return df


# ============================================================
# LOAD DATA
# ============================================================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")
print()


# ============================================================
# SELECT REQUIRED COLUMNS
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


# ============================================================
# CLEAN DATA
# ============================================================

df = df.dropna()

df = df[
    (df["selling_price"] > 0)
    & (df["km_driven"] >= 0)
    & (df["year"] >= 1990)
    & (df["year"] <= CURRENT_YEAR)
]

print(f"Clean dataset shape: {df.shape}")
print()


# ============================================================
# FEATURES / TARGET
# ============================================================

X = df.drop(columns=["selling_price"])
y = df["selling_price"]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")
print()


# ============================================================
# PREPROCESSING
# ============================================================

categorical_features = [
    "name",
    "brand",
    "fuel",
    "seller_type",
    "transmission",
    "owner",
]

numerical_features = [
    "year",
    "km_driven",
    "car_age",
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=2,
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
# MODELS
# ============================================================

models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        max_features="sqrt",
    ),

    "Extra Trees": ExtraTreesRegressor(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        max_features=1.0,
    ),
}


# ============================================================
# EXPERIMENTS
# ============================================================

results = []

best_model = None
best_model_name = None
best_r2 = float("-inf")


for model_name, regressor in models.items():

    for target_type in ["normal", "log"]:

        print("=" * 60)
        print(f"Experiment: {model_name} + {target_type} target")
        print("=" * 60)

        # Feature engineering is inside the pipeline.
        # Therefore API input remains unchanged.
        base_pipeline = Pipeline(
            steps=[
                (
                    "feature_engineering",
                    FunctionTransformer(
                        add_features,
                        validate=False,
                    ),
                ),
                (
                    "preprocessor",
                    preprocessor,
                ),
                (
                    "model",
                    regressor,
                ),
            ]
        )

        # Log-transform target for some experiments
        if target_type == "log":

            pipeline = TransformedTargetRegressor(
                regressor=base_pipeline,
                func=np.log1p,
                inverse_func=np.expm1,
            )

        else:
            pipeline = base_pipeline

        # Train
        pipeline.fit(X_train, y_train)

        # Predict
        predictions = pipeline.predict(X_test)

        # Metrics
        mae = mean_absolute_error(
            y_test,
            predictions,
        )

        r2 = r2_score(
            y_test,
            predictions,
        )

        print(f"MAE: ₹{mae:,.2f}")
        print(f"R² : {r2:.4f}")
        print()

        results.append(
            {
                "model": model_name,
                "target": target_type,
                "MAE": mae,
                "R2": r2,
            }
        )

        # Keep best model
        if r2 > best_r2:
            best_r2 = r2
            best_model = pipeline
            best_model_name = f"{model_name} + {target_type}"


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="R2",
    ascending=False,
)

print()
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        formatters={
            "MAE": lambda x: f"₹{x:,.2f}",
            "R2": lambda x: f"{x:.4f}",
        },
    )
)

print()
print("=" * 70)
print("BEST MODEL")
print("=" * 70)

print(f"Model: {best_model_name}")
print(f"R²   : {best_r2:.4f}")
print()


# ============================================================
# SAVE BEST MODEL
# ============================================================

MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

joblib.dump(
    best_model,
    MODEL_PATH,
)

print(f"Saved best model to:")
print(MODEL_PATH)