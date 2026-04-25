import ast
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


QUALITY_MAP = {
    0: "bad",
    1: "normal",
    2: "good",
}


def safe_literal_eval(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, list):
        return x
    try:
        return ast.literal_eval(x)
    except Exception:
        return np.nan


def parse_nutrition(x):
    val = safe_literal_eval(x)
    if isinstance(val, list) and len(val) >= 7:
        return val[:7]
    return [np.nan] * 7


def list_len(x):
    val = safe_literal_eval(x)
    if isinstance(val, list):
        return len(val)
    return np.nan


def avg_item_len(x):
    val = safe_literal_eval(x)
    if isinstance(val, list) and len(val) > 0:
        lengths = [len(str(i)) for i in val]
        return float(np.mean(lengths))
    return np.nan


def total_text_len_from_list(x):
    val = safe_literal_eval(x)
    if isinstance(val, list):
        return sum(len(str(i)) for i in val)
    return np.nan


def missing_report(df, name="df"):
    rep = pd.DataFrame({
        "missing_cnt": df.isna().sum(),
        "missing_pct": (df.isna().mean() * 100).round(2),
        "dtype": df.dtypes.astype(str),
    }).sort_values("missing_pct", ascending=False)

    print(f"\n=== Missing report: {name} ===")
    rep_nonzero = rep[rep["missing_cnt"] > 0]
    if len(rep_nonzero) == 0:
        print("Пропусков нет.")
    else:
        print(rep_nonzero.head(30).to_string())
    return rep


def iqr_bounds(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    return low, high


def outlier_report(df, numeric_cols):
    rows = []
    for col in numeric_cols:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) == 0:
            continue
        low, high = iqr_bounds(s)
        cnt = ((s < low) | (s > high)).sum()
        rows.append([col, len(s), cnt, round(cnt / len(s) * 100, 2), low, high])

    rep = pd.DataFrame(
        rows,
        columns=["feature", "n", "outliers_cnt", "outliers_pct", "low", "high"],
    )
    return rep.sort_values("outliers_pct", ascending=False)


def load_raw_data(data_dir="data/raw"):
    data_dir = Path(data_dir)
    recipes = pd.read_csv(data_dir / "RAW_recipes.csv")
    interactions = pd.read_csv(data_dir / "RAW_interactions.csv")
    return recipes, interactions


def print_initial_overview(recipes, interactions):
    print("recipes shape:", recipes.shape)
    print("interactions shape:", interactions.shape)

    print("\nПервые 3 строки recipes:")
    print(recipes.head(3).to_string())

    print("\nПервые 3 строки interactions:")
    print(interactions.head(3).to_string())

    print("RECIPES INFO")
    print(recipes.info())

    print("\nINTERACTIONS INFO")
    print(interactions.info())

    print("\nrecipes columns:")
    print(recipes.columns.tolist())

    print("\ninteractions columns:")
    print(interactions.columns.tolist())


def clean_recipes(recipes):
    recipes = recipes.copy()

    if "id" in recipes.columns:
        recipes = recipes.rename(columns={"id": "recipe_id"})

    print("recipes duplicates by full row:", recipes.duplicated().sum())
    print("recipes duplicates by recipe_id:", recipes.duplicated(subset=["recipe_id"]).sum())

    recipes = recipes.drop_duplicates()
    recipes = recipes.drop_duplicates(subset=["recipe_id"], keep="first")

    for col in ["minutes", "n_steps", "n_ingredients", "contributor_id", "recipe_id"]:
        if col in recipes.columns:
            recipes[col] = pd.to_numeric(recipes[col], errors="coerce")

    if "submitted" in recipes.columns:
        recipes["submitted"] = pd.to_datetime(recipes["submitted"], errors="coerce")

    recipes = recipes[recipes["recipe_id"].notna()]
    recipes = recipes[recipes["minutes"].notna()]
    recipes = recipes[recipes["n_steps"].notna()]
    recipes = recipes[recipes["n_ingredients"].notna()]

    recipes = recipes[recipes["minutes"] > 0]
    recipes = recipes[recipes["n_steps"] > 0]
    recipes = recipes[recipes["n_ingredients"] > 0]

    return recipes


def clean_interactions(interactions):
    interactions = interactions.copy()

    print("interactions duplicates by full row:", interactions.duplicated().sum())
    interactions = interactions.drop_duplicates()

    for col in ["user_id", "recipe_id", "rating"]:
        if col in interactions.columns:
            interactions[col] = pd.to_numeric(interactions[col], errors="coerce")

    if "date" in interactions.columns:
        interactions["date"] = pd.to_datetime(interactions["date"], errors="coerce")

    interactions = interactions[interactions["recipe_id"].notna()]
    interactions = interactions[interactions["rating"].between(1, 5, inclusive="both")]

    return interactions


def build_recipe_target(recipes, interactions, min_rating_count=10):
    rating_stats = (
        interactions
        .groupby("recipe_id", as_index=False)
        .agg(
            mean_rating=("rating", "mean"),
            median_rating=("rating", "median"),
            rating_count=("rating", "size"),
        )
    )

    data = recipes.merge(rating_stats, on="recipe_id", how="left")
    data = data[data["rating_count"] >= min_rating_count].copy()

    q30 = data["mean_rating"].quantile(0.30)
    q70 = data["mean_rating"].quantile(0.70)

    print("raw thresholds:", q30, q70)
    print("rounded for report:", round(q30, 1), round(q70, 1))

    conditions = [
        data["mean_rating"] <= q30,
        data["mean_rating"] >= q70,
    ]
    choices = [0, 2]

    data["recipe_quality"] = np.select(conditions, choices, default=1).astype(int)
    data["recipe_quality_name"] = data["recipe_quality"].map(QUALITY_MAP)

    print("final recipe-level dataset shape:", data.shape)

    print("\nРаспределение классов:")
    print(data["recipe_quality_name"].value_counts().to_string())

    print("\nДоли классов:")
    print(data["recipe_quality_name"].value_counts(normalize=True).round(4).to_string())

    print("\nПроверка диапазонов mean_rating по классам:")
    print(
        data.groupby("recipe_quality_name")["mean_rating"]
        .agg(["count", "min", "max", "mean"])
        .round(4)
        .to_string()
    )

    return data


def engineer_features(data):
    data = data.copy()

    for col in ["tags", "steps", "ingredients"]:
        if col in data.columns:
            data[f"{col}_len"] = data[col].apply(list_len)
            data[f"{col}_avg_item_len"] = data[col].apply(avg_item_len)
            data[f"{col}_total_text_len"] = data[col].apply(total_text_len_from_list)

    if "name" in data.columns:
        data["name_len"] = data["name"].fillna("").astype(str).str.len()
        data["name_word_count"] = data["name"].fillna("").astype(str).str.split().str.len()

    if "description" in data.columns:
        data["description_len"] = data["description"].fillna("").astype(str).str.len()
        data["description_word_count"] = data["description"].fillna("").astype(str).str.split().str.len()

    if "submitted" in data.columns:
        data["submitted_year"] = data["submitted"].dt.year
        data["submitted_month"] = data["submitted"].dt.month
        data["submitted_weekday"] = data["submitted"].dt.weekday

    nutrition_cols = [
        "calories",
        "total_fat_pdv",
        "sugar_pdv",
        "sodium_pdv",
        "protein_pdv",
        "saturated_fat_pdv",
        "carbs_pdv",
    ]

    nutrition_df = pd.DataFrame(
        data["nutrition"].apply(parse_nutrition).tolist(),
        columns=nutrition_cols,
        index=data.index,
    )

    data = pd.concat([data, nutrition_df], axis=1)

    data["minutes_per_step"] = data["minutes"] / data["n_steps"]
    data["ingredients_per_step"] = data["n_ingredients"] / data["n_steps"]
    data["calories_per_ingredient"] = data["calories"] / data["n_ingredients"]
    data["calories_per_step"] = data["calories"] / data["n_steps"]

    data["is_quick_recipe"] = (data["minutes"] <= 30).astype(int)
    data["is_long_recipe"] = (data["minutes"] >= 120).astype(int)
    data["is_few_ingredients"] = (data["n_ingredients"] <= 7).astype(int)

    for col in ["minutes", "description_len", "steps_total_text_len", "ingredients_total_text_len", "calories"]:
        if col in data.columns:
            data[f"log1p_{col}"] = np.log1p(data[col])

    print("Количество столбцов после feature engineering:", data.shape[1])
    print(data.head(3).to_string())

    return data


def make_outlier_report(data):
    numeric_for_outliers = [
        "minutes", "n_steps", "n_ingredients",
        "calories", "total_fat_pdv", "sugar_pdv", "sodium_pdv",
        "protein_pdv", "saturated_fat_pdv", "carbs_pdv",
        "description_len", "steps_len", "ingredients_len",
        "minutes_per_step", "ingredients_per_step", "calories_per_ingredient",
    ]
    numeric_for_outliers = [c for c in numeric_for_outliers if c in data.columns]
    outliers_df = outlier_report(data, numeric_for_outliers)
    print(outliers_df.head(20).to_string(index=False))
    return outliers_df


def get_feature_columns(data):
    base_feature_cols = [
        "minutes",
        "n_steps",
        "n_ingredients",
        "calories",
        "total_fat_pdv",
        "sugar_pdv",
        "sodium_pdv",
        "protein_pdv",
        "saturated_fat_pdv",
        "carbs_pdv",
    ]

    full_feature_cols = [
        "minutes",
        "n_steps",
        "n_ingredients",
        "tags_len",
        "steps_len",
        "ingredients_len",
        "tags_avg_item_len",
        "steps_avg_item_len",
        "ingredients_avg_item_len",
        "tags_total_text_len",
        "steps_total_text_len",
        "ingredients_total_text_len",
        "name_len",
        "name_word_count",
        "description_len",
        "description_word_count",
        "submitted_year",
        "submitted_month",
        "submitted_weekday",
        "calories",
        "total_fat_pdv",
        "sugar_pdv",
        "sodium_pdv",
        "protein_pdv",
        "saturated_fat_pdv",
        "carbs_pdv",
        "minutes_per_step",
        "ingredients_per_step",
        "calories_per_ingredient",
        "calories_per_step",
        "is_quick_recipe",
        "is_long_recipe",
        "is_few_ingredients",
        "log1p_minutes",
        "log1p_description_len",
        "log1p_steps_total_text_len",
        "log1p_ingredients_total_text_len",
        "log1p_calories",
    ]

    base_feature_cols = [c for c in base_feature_cols if c in data.columns]
    full_feature_cols = [c for c in full_feature_cols if c in data.columns]

    return base_feature_cols, full_feature_cols


def prepare_feature_sets(data):
    base_feature_cols, full_feature_cols = get_feature_columns(data)

    X_base = data[base_feature_cols].copy()
    X_full = data[full_feature_cols].copy()
    y = data["recipe_quality"].copy()

    print("X_base shape:", X_base.shape)
    print("base feature count:", len(base_feature_cols))
    print("base features:", base_feature_cols)

    print("\nX_full shape:", X_full.shape)
    print("full feature count:", len(full_feature_cols))
    print("Распределение y:")
    print(y.value_counts().sort_index().to_string())

    return X_base, X_full, y, base_feature_cols, full_feature_cols


def make_splits(X_base, X_full, y, random_state=42):
    indices = np.arange(len(y))

    idx_train, idx_temp, y_train, y_temp = train_test_split(
        indices,
        y,
        test_size=0.30,
        random_state=random_state,
        stratify=y,
    )

    idx_val, idx_test, y_val, y_test = train_test_split(
        idx_temp,
        y_temp,
        test_size=0.50,
        random_state=random_state,
        stratify=y_temp,
    )

    X_train_base = X_base.iloc[idx_train].copy()
    X_val_base = X_base.iloc[idx_val].copy()
    X_test_base = X_base.iloc[idx_test].copy()

    X_train_full = X_full.iloc[idx_train].copy()
    X_val_full = X_full.iloc[idx_val].copy()
    X_test_full = X_full.iloc[idx_test].copy()

    print("Train base:", X_train_base.shape)
    print(y_train.value_counts(normalize=True).sort_index().round(4).to_string())

    print("\nVal base:", X_val_base.shape)
    print(y_val.value_counts(normalize=True).sort_index().round(4).to_string())

    print("\nTest base:", X_test_base.shape)
    print(y_test.value_counts(normalize=True).sort_index().round(4).to_string())

    print("\nTrain full:", X_train_full.shape)
    print("Val full:", X_val_full.shape)
    print("Test full:", X_test_full.shape)

    return {
        "X_train_base": X_train_base,
        "X_val_base": X_val_base,
        "X_test_base": X_test_base,
        "X_train_full": X_train_full,
        "X_val_full": X_val_full,
        "X_test_full": X_test_full,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }


def save_processed_artifacts(
    data,
    recipes,
    interactions,
    y,
    processed_dir="data/processed",
):
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    summary_table = pd.DataFrame({
        "rows": [data.shape[0]],
        "columns": [data.shape[1]],
        "class_0_bad_share": [round((y == 0).mean(), 4)],
        "class_1_normal_share": [round((y == 1).mean(), 4)],
        "class_2_good_share": [round((y == 2).mean(), 4)],
    })

    cleaning_table = pd.DataFrame({
        "recipes_rows_after_cleaning": [recipes.shape[0]],
        "interactions_rows_after_cleaning": [interactions.shape[0]],
        "recipe_level_rows_after_target": [data.shape[0]],
    })

    data.to_csv(processed_dir / "recipes_with_quality_classes.csv", index=False, encoding="utf-8")
    summary_table.to_csv(processed_dir / "summary_table.csv", index=False, encoding="utf-8")
    cleaning_table.to_csv(processed_dir / "cleaning_table.csv", index=False, encoding="utf-8")

    return summary_table, cleaning_table