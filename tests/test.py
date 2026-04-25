from src.preprocessing import (
    load_raw_data,
    clean_recipes,
    clean_interactions,
    build_recipe_target,
    engineer_features,
    prepare_feature_sets,
    make_splits,
)
from src.modeling import make_dummy


def test_pipeline_smoke():
    recipes, interactions = load_raw_data("data/raw")

    recipes = clean_recipes(recipes)
    interactions = clean_interactions(interactions)

    assert len(recipes) > 0
    assert len(interactions) > 0

    data = build_recipe_target(recipes, interactions, min_rating_count=10)
    data = engineer_features(data)

    assert "recipe_quality" in data.columns
    assert "recipe_quality_name" in data.columns

    X_base, X_full, y, base_feature_cols, full_feature_cols = prepare_feature_sets(data)

    forbidden = {"mean_rating", "median_rating", "rating_count", "recipe_quality", "recipe_quality_name"}
    assert forbidden.isdisjoint(set(base_feature_cols))
    assert forbidden.isdisjoint(set(full_feature_cols))

    splits = make_splits(X_base, X_full, y, random_state=42)

    model = make_dummy()
    model.fit(splits["X_train_base"], splits["y_train"])
    pred = model.predict(splits["X_val_base"])

    assert len(pred) == len(splits["y_val"])