from src.preprocessing import (
    load_raw_data,
    print_initial_overview,
    clean_recipes,
    clean_interactions,
    missing_report,
    build_recipe_target,
    engineer_features,
    make_outlier_report,
    prepare_feature_sets,
    make_splits,
    save_processed_artifacts,
)
from src.modeling import (
    run_all_experiments,
    evaluate_best_model,
    save_model_artifacts,
)


def main():
    recipes, interactions = load_raw_data("data/raw")

    print_initial_overview(recipes, interactions)

    recipes = clean_recipes(recipes)
    interactions = clean_interactions(interactions)

    print("recipes shape after cleaning:", recipes.shape)
    print("interactions shape after cleaning:", interactions.shape)

    missing_report(recipes, "recipes")
    missing_report(interactions, "interactions")

    data = build_recipe_target(recipes, interactions, min_rating_count=10)
    data = engineer_features(data)
    make_outlier_report(data)

    X_base, X_full, y, base_feature_cols, full_feature_cols = prepare_feature_sets(data)
    splits = make_splits(X_base, X_full, y, random_state=42)

    summary_table, cleaning_table = save_processed_artifacts(
        data=data,
        recipes=recipes,
        interactions=interactions,
        y=y,
        processed_dir="data/processed",
    )

    print("\nsummary_table:")
    print(summary_table.to_string(index=False))

    print("\ncleaning_table:")
    print(cleaning_table.to_string(index=False))

    experiments, results_df, best_by_family_df = run_all_experiments(splits)

    print("\nИтоговая таблица экспериментов на validation:")
    print(results_df.to_string(index=False))

    print("\nЛучшие модели по семействам:")
    print(best_by_family_df.to_string(index=False))

    best_exp, best_model_fitted, best_feature_cols = evaluate_best_model(
        experiments=experiments,
        results_df=results_df,
        splits=splits,
        base_feature_cols=base_feature_cols,
        full_feature_cols=full_feature_cols,
    )

    print("\n15 лучших экспериментов:")
    print(results_df.head(15).to_string(index=False))

    save_model_artifacts(
        results_df=results_df,
        best_by_family_df=best_by_family_df,
        best_model_fitted=best_model_fitted,
        best_feature_cols=best_feature_cols,
        processed_dir="data/processed",
        models_dir="models",
    )


if __name__ == "__main__":
    main()