from pathlib import Path

import joblib
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier

from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)


TARGET_NAMES = ["bad", "normal", "good"]
TARGET_LABELS = [0, 1, 2]


def make_dummy():
    return DummyClassifier(strategy="most_frequent")


def make_logreg(C=1.0, use_pca=False, pca_n_components=0.95):
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler())
    ]
    if use_pca:
        steps.append(("pca", PCA(n_components=pca_n_components)))
    steps.append((
        "model",
        LogisticRegression(
            C=C,
            max_iter=15000,
            tol=1e-3,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42
        )
    ))
    return Pipeline(steps=steps)


def make_knn(n_neighbors=25, weights="distance", use_pca=False, pca_n_components=0.95):
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ]
    if use_pca:
        steps.append(("pca", PCA(n_components=pca_n_components)))
    steps.append((
        "model",
        KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            n_jobs=-1,
        )
    ))
    return Pipeline(steps=steps)


def make_rf(max_depth=20, min_samples_leaf=5, n_estimators=300, max_features="sqrt"):
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )),
    ])


def make_extra_trees(max_depth=20, min_samples_leaf=5, n_estimators=400, max_features="sqrt"):
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("model", ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )),
    ])


def make_hgb(learning_rate=0.05, max_depth=6, max_iter=300, min_samples_leaf=20):
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingClassifier(
            learning_rate=learning_rate,
            max_depth=max_depth,
            max_iter=max_iter,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
        )),
    ])


def calc_metrics(y_true, y_pred):
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }


def stringify_params(params):
    if not params:
        return "-"
    return ", ".join([f"{k}={v}" for k, v in params.items()])


def run_experiment(name, family, model, feature_set, X_tr, y_tr, X_ev, y_ev, params=None):
    fitted_model = clone(model)
    fitted_model.fit(X_tr, y_tr)
    pred = fitted_model.predict(X_ev)

    metrics = calc_metrics(y_ev, pred)
    return {
        "experiment": name,
        "family": family,
        "feature_set": feature_set,
        "n_features": X_tr.shape[1],
        "params": stringify_params(params),
        "macro_f1": round(metrics["macro_f1"], 4),
        "weighted_f1": round(metrics["weighted_f1"], 4),
        "balanced_accuracy": round(metrics["balanced_accuracy"], 4),
    }


def build_experiments():
    experiments = []

    experiments.append({
        "name": "Dummy_base",
        "family": "Dummy",
        "model": make_dummy(),
        "feature_set": "base",
        "params": {},
    })

    for params in ParameterGrid({"C": [0.3, 1.0, 3.0]}):
        experiments.append({
            "name": f"LogReg_base_C{params['C']}",
            "family": "LogReg",
            "model": make_logreg(C=params["C"], use_pca=False),
            "feature_set": "base",
            "params": params,
        })
        experiments.append({
            "name": f"LogReg_full_C{params['C']}",
            "family": "LogReg",
            "model": make_logreg(C=params["C"], use_pca=False),
            "feature_set": "full",
            "params": params,
        })

    for params in ParameterGrid({
        "n_neighbors": [9, 15, 25, 35],
        "weights": ["uniform", "distance"],
    }):
        experiments.append({
            "name": f"KNN_base_k{params['n_neighbors']}_{params['weights']}",
            "family": "KNN",
            "model": make_knn(
                n_neighbors=params["n_neighbors"],
                weights=params["weights"],
                use_pca=False,
            ),
            "feature_set": "base",
            "params": params,
        })
        experiments.append({
            "name": f"KNN_full_k{params['n_neighbors']}_{params['weights']}",
            "family": "KNN",
            "model": make_knn(
                n_neighbors=params["n_neighbors"],
                weights=params["weights"],
                use_pca=False,
            ),
            "feature_set": "full",
            "params": params,
        })

    for params in ParameterGrid({
        "max_depth": [10, 20, None],
        "min_samples_leaf": [3, 5, 10],
        "max_features": ["sqrt", 0.7],
    }):
        experiments.append({
            "name": f"RF_full_d{params['max_depth']}_l{params['min_samples_leaf']}_mf{params['max_features']}",
            "family": "RandomForest",
            "model": make_rf(
                max_depth=params["max_depth"],
                min_samples_leaf=params["min_samples_leaf"],
                n_estimators=300,
                max_features=params["max_features"],
            ),
            "feature_set": "full",
            "params": params,
        })

    for params in ParameterGrid({
        "max_depth": [10, 20, None],
        "min_samples_leaf": [3, 5],
        "max_features": ["sqrt", 0.7],
    }):
        experiments.append({
            "name": f"ET_full_d{params['max_depth']}_l{params['min_samples_leaf']}_mf{params['max_features']}",
            "family": "ExtraTrees",
            "model": make_extra_trees(
                max_depth=params["max_depth"],
                min_samples_leaf=params["min_samples_leaf"],
                n_estimators=400,
                max_features=params["max_features"],
            ),
            "feature_set": "full",
            "params": params,
        })

    for params in ParameterGrid({
        "learning_rate": [0.03, 0.05, 0.1],
        "max_depth": [4, 6, 8],
        "min_samples_leaf": [20, 40],
    }):
        experiments.append({
            "name": f"HGB_full_lr{params['learning_rate']}_d{params['max_depth']}_l{params['min_samples_leaf']}",
            "family": "HistGB",
            "model": make_hgb(
                learning_rate=params["learning_rate"],
                max_depth=params["max_depth"],
                max_iter=300,
                min_samples_leaf=params["min_samples_leaf"],
            ),
            "feature_set": "full",
            "params": params,
        })

    for params in ParameterGrid({
        "C": [1.0, 3.0],
        "pca_n_components": [5, 10, 0.95],
    }):
        experiments.append({
            "name": f"LogReg_full_PCA_{params['pca_n_components']}_C{params['C']}",
            "family": "LogReg_PCA",
            "model": make_logreg(
                C=params["C"],
                use_pca=True,
                pca_n_components=params["pca_n_components"],
            ),
            "feature_set": "full",
            "params": params,
        })

    for params in ParameterGrid({
        "n_neighbors": [15, 25],
        "weights": ["distance"],
        "pca_n_components": [5, 10, 0.95],
    }):
        experiments.append({
            "name": f"KNN_full_PCA_{params['pca_n_components']}_k{params['n_neighbors']}",
            "family": "KNN_PCA",
            "model": make_knn(
                n_neighbors=params["n_neighbors"],
                weights=params["weights"],
                use_pca=True,
                pca_n_components=params["pca_n_components"],
            ),
            "feature_set": "full",
            "params": params,
        })

    return experiments


def run_all_experiments(splits):
    X_train_base = splits["X_train_base"]
    X_val_base = splits["X_val_base"]
    X_train_full = splits["X_train_full"]
    X_val_full = splits["X_val_full"]
    y_train = splits["y_train"]
    y_val = splits["y_val"]

    experiments = build_experiments()
    print("Всего экспериментов:", len(experiments))

    all_results = []

    for i, exp in enumerate(experiments, start=1):
        if exp["feature_set"] == "base":
            X_tr = X_train_base
            X_ev = X_val_base
        else:
            X_tr = X_train_full
            X_ev = X_val_full

        result = run_experiment(
            name=exp["name"],
            family=exp["family"],
            model=exp["model"],
            feature_set=exp["feature_set"],
            X_tr=X_tr,
            y_tr=y_train,
            X_ev=X_ev,
            y_ev=y_val,
            params=exp["params"],
        )
        all_results.append(result)

    results_df = pd.DataFrame(all_results).sort_values(
        ["macro_f1", "balanced_accuracy", "weighted_f1"],
        ascending=False,
    ).reset_index(drop=True)

    best_by_family_df = (
        results_df
        .sort_values(["family", "macro_f1", "balanced_accuracy"], ascending=[True, False, False])
        .groupby("family", as_index=False)
        .head(1)
        .sort_values(["macro_f1", "balanced_accuracy"], ascending=False)
        .reset_index(drop=True)
    )

    return experiments, results_df, best_by_family_df


def print_detailed_report(model, X_tr, y_tr, X_te, y_te, model_name):
    fitted_model = clone(model)
    fitted_model.fit(X_tr, y_tr)
    pred = fitted_model.predict(X_te)

    macro_f1 = f1_score(y_te, pred, average="macro")
    weighted_f1 = f1_score(y_te, pred, average="weighted")
    bal_acc = balanced_accuracy_score(y_te, pred)

    print(f"\n{model_name} ")
    print("Macro F1:", round(macro_f1, 4))
    print("Weighted F1:", round(weighted_f1, 4))
    print("Balanced accuracy:", round(bal_acc, 4))

    print("\nClassification report:")
    print(classification_report(
        y_te,
        pred,
        labels=TARGET_LABELS,
        target_names=TARGET_NAMES,
        digits=4,
    ))

    cm = confusion_matrix(y_te, pred, labels=TARGET_LABELS)
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{x}" for x in TARGET_NAMES],
        columns=[f"pred_{x}" for x in TARGET_NAMES],
    )
    print("Confusion matrix:")
    print(cm_df.to_string())

    return fitted_model


def evaluate_best_model(experiments, results_df, splits, base_feature_cols, full_feature_cols):
    best_experiment_name = results_df.iloc[0]["experiment"]
    print("BEST EXPERIMENT ON VAL:", best_experiment_name)

    best_exp = None
    for exp in experiments:
        if exp["name"] == best_experiment_name:
            best_exp = exp
            break

    if best_exp is None:
        raise ValueError("Лучшая модель не найдена в experiments")

    if best_exp["feature_set"] == "base":
        X_train_final = pd.concat([splits["X_train_base"], splits["X_val_base"]], axis=0)
        X_test_final = splits["X_test_base"]
        best_feature_cols = base_feature_cols
    else:
        X_train_final = pd.concat([splits["X_train_full"], splits["X_val_full"]], axis=0)
        X_test_final = splits["X_test_full"]
        best_feature_cols = full_feature_cols

    y_train_final = pd.concat([splits["y_train"], splits["y_val"]], axis=0)

    best_model_fitted = print_detailed_report(
        model=best_exp["model"],
        X_tr=X_train_final,
        y_tr=y_train_final,
        X_te=X_test_final,
        y_te=splits["y_test"],
        model_name=f"{best_experiment_name}_TEST",
    )

    print("\nЛучшая модель:")
    print("experiment:", best_exp["name"])
    print("family:", best_exp["family"])
    print("feature_set:", best_exp["feature_set"])
    print("params:", best_exp["params"])

    return best_exp, best_model_fitted, best_feature_cols


def save_model_artifacts(
    results_df,
    best_by_family_df,
    best_model_fitted,
    best_feature_cols,
    processed_dir="data/processed",
    models_dir="models",
):
    processed_dir = Path(processed_dir)
    models_dir = Path(models_dir)

    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(processed_dir / "experiments_validation_results.csv", index=False, encoding="utf-8")
    best_by_family_df.to_csv(processed_dir / "best_models_by_family.csv", index=False, encoding="utf-8")

    pd.Series(best_feature_cols, name="feature_name").to_csv(
        models_dir / "best_model_features.csv",
        index=False,
        encoding="utf-8",
    )

    joblib.dump(best_model_fitted, models_dir / "best_model.joblib")