from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance

from src.modeling import calc_metrics


def build_feature_groups(full_feature_cols):
    nutrition_cols = {
        "calories",
        "total_fat_pdv",
        "sugar_pdv",
        "sodium_pdv",
        "protein_pdv",
        "saturated_fat_pdv",
        "carbs_pdv",
        "calories_per_ingredient",
        "calories_per_step",
        "log1p_calories",
    }

    text_structure_cols = {
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
        "log1p_description_len",
        "log1p_steps_total_text_len",
        "log1p_ingredients_total_text_len",
    }

    date_cols = {
        "submitted_year",
        "submitted_month",
        "submitted_weekday",
    }

    complexity_cols = {
        "minutes",
        "n_steps",
        "n_ingredients",
        "minutes_per_step",
        "ingredients_per_step",
        "is_quick_recipe",
        "is_long_recipe",
        "is_few_ingredients",
        "log1p_minutes",
    }

    full_set = set(full_feature_cols)

    feature_groups = {
        "full": list(full_feature_cols),
        "without_nutrition": [c for c in full_feature_cols if c not in nutrition_cols],
        "without_text_structure": [c for c in full_feature_cols if c not in text_structure_cols],
        "without_date": [c for c in full_feature_cols if c not in date_cols],
        "complexity_only": [c for c in full_feature_cols if c in complexity_cols],
        "nutrition_only": [c for c in full_feature_cols if c in nutrition_cols],
        "text_structure_only": [c for c in full_feature_cols if c in text_structure_cols],
    }

    feature_groups = {
        name: cols
        for name, cols in feature_groups.items()
        if len(cols) > 0 and set(cols).issubset(full_set)
    }

    return feature_groups


def run_ablation_study(model, splits, feature_groups):
    results = []

    X_train_full = splits["X_train_full"]
    X_val_full = splits["X_val_full"]
    y_train = splits["y_train"]
    y_val = splits["y_val"]

    for group_name, cols in feature_groups.items():
        fitted_model = clone(model)

        fitted_model.fit(X_train_full[cols], y_train)
        pred = fitted_model.predict(X_val_full[cols])

        metrics = calc_metrics(y_val, pred)

        results.append(
            {
                "feature_group": group_name,
                "n_features": len(cols),
                "macro_f1": round(metrics["macro_f1"], 4),
                "weighted_f1": round(metrics["weighted_f1"], 4),
                "balanced_accuracy": round(metrics["balanced_accuracy"], 4),
            }
        )

    result_df = pd.DataFrame(results).sort_values(
        ["macro_f1", "balanced_accuracy", "weighted_f1"],
        ascending=False,
    ).reset_index(drop=True)

    return result_df


def get_tree_feature_importance(model, feature_cols, top_n=20):
    if not hasattr(model, "named_steps"):
        raise ValueError("Expected sklearn Pipeline with named_steps.")

    tree_model = model.named_steps.get("model")

    if tree_model is None or not hasattr(tree_model, "feature_importances_"):
        raise ValueError("Final estimator does not have feature_importances_.")

    importance_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": tree_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False).reset_index(drop=True)

    return importance_df.head(top_n)


def get_permutation_importance_df(
    model,
    X,
    y,
    feature_cols,
    top_n=20,
    n_repeats=5,
    random_state=42,
):
    result = permutation_importance(
        model,
        X,
        y,
        scoring="f1_macro",
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    importance_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    return importance_df.head(top_n)


def plot_top_features(df, feature_col, value_col, title, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_df = df.sort_values(value_col, ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(plot_df[feature_col], plot_df[value_col])
    plt.title(title)
    plt.xlabel(value_col)
    plt.ylabel("feature")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show()