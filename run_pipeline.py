import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.base import clone
from src.preprocessing import get_prepared_data
from src.statistical_tests import run_statistical_tests
from sklearn.metrics import ( roc_auc_score, accuracy_score, precision_score, recall_score, f1_score)

from src.evaluation import ( 
    plot_feature_importance, plot_confusion_matrices, 
    plot_roc_curves, plot_cv_boxplot, plot_logistic_regression_coefs
)
from src.models import get_models

# Cenário A = sem a coluna "Have you ever had suicidal thoughts ?" (evita data leakage)
# Cenário B = com a coluna (apenas para demonstrar o efeito do leakage)

MODEL_DIR = Path("model_artifacts")
MODEL_DIR.mkdir(exist_ok=True)

def run_pipeline():
    print("Iniciando o Pipeline de Machine Learning...")

    # ------------------------------------------------------------------
    # PARTE 1: CENÁRIO A — Projeto oficial (sem vazamento)
    # ------------------------------------------------------------------
    print("\n[1/4] Carregando e pré-processando dados (Cenário A)...")
    X_train_A, X_test_A, y_train_A, y_test_A, feature_names_A = get_prepared_data(cenario='A')

    print("\n[2/4] Treinando modelos...")
    models = get_models()

    n_splits = 10
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    cv_raw = {}
    trained_models = {}
    cv_summary_rows = []

    print(f"\nResultados da Validação Cruzada ( {n_splits} Folds):")

    for name, pipe in models.items():
        res = cross_validate(clone(pipe), X_train_A, y_train_A, cv=cv, scoring=scoring, n_jobs=-1)
        cv_raw[name] = res

        row = {"Modelo": name}
        for s in scoring:
            row[f"{s}_mean"] = res[f"test_{s}"].mean()
            row[f"{s}_std"]  = res[f"test_{s}"].std()
        cv_summary_rows.append(row)

        print(f"  {name:22s} | acc={row['accuracy_mean']:.4f}  f1={row['f1_mean']:.4f}  auc={row['roc_auc_mean']:.4f}")

        pipe.fit(X_train_A, y_train_A)
        trained_models[name] = pipe

    cv_summary = (
        pd.DataFrame(cv_summary_rows)
        .sort_values("f1_mean", ascending=False)
        .reset_index(drop=True)
    )
    print("\nResumo (ordenado por F1):")
    print(cv_summary[["Modelo", "f1_mean", "accuracy_mean", "roc_auc_mean"]])

    print("\n Avaliação final no conjunto de teste...")

    resultados_teste = []
    y_scores = {}

    for name, pipe in trained_models.items():
        y_pred = pipe.predict(X_test_A)

        y_score = (
            pipe.predict_proba(X_test_A)[:, 1]
            if hasattr(pipe, "predict_proba")
            else pipe.decision_function(X_test_A)
        )

        y_scores[name] = y_score

        resultados_teste.append({
            "Modelo":    name,
            "accuracy":  accuracy_score(y_test_A, y_pred),
            "precision": precision_score(y_test_A, y_pred),
            "recall":    recall_score(y_test_A, y_pred),
            "f1":        f1_score(y_test_A, y_pred),
            "roc_auc":   roc_auc_score(y_test_A, y_score),
        })

    df_teste = (
        pd.DataFrame(resultados_teste)
        .sort_values("f1", ascending=False)
        .reset_index(drop=True)
    )

    print(df_teste.round(4).to_string(index=False))
    df_teste.to_csv("results/resultados_teste.csv", index=False)
    print("=> Salvo em 'results/resultados_teste.csv'")

    print("\n[3/4] Avaliando modelos e gerando visualizações...")

    plot_confusion_matrices(trained_models, X_test_A, y_test_A)
    plot_roc_curves(trained_models, X_test_A, y_test_A)
    plot_cv_boxplot(cv_raw)

    # Regressão Logística: acessa o estimador dentro do Pipeline
    logreg_estimator = trained_models["Regressão Logística"].named_steps["model"]
    plot_logistic_regression_coefs(logreg_estimator, feature_names_A)

    if "Random Forest" in trained_models:
        rf_estimator = trained_models["Random Forest"].named_steps["model"]
        plot_feature_importance(rf_estimator, feature_names_A, "rf_importances_cenarioA.png")

    print("\n[4/4] Rodando testes estatísticos (Friedman + Wilcoxon/Bonferroni)...")

    df_statistics = run_statistical_tests(cv_raw, metric_for_test="test_f1")
    df_statistics.to_csv("results/tabela_wilcoxon_bonferroni.csv", index=False)
    print("=> Salvo em 'results/tabela_wilcoxon_bonferroni.csv'")

    bundle_A = {
        "modelos":       trained_models,
        "feature_order": feature_names_A,
        "degree_cols":   [c for c in feature_names_A if c.startswith("Degree_")],
        "tem_leak":      False,
    }
    joblib.dump(bundle_A, MODEL_DIR / "bundle_sem_leak.pkl")
    print("\n=> bundle_sem_leak.pkl salvo em model_artifacts/")

    # ------------------------------------------------------------------
    # PARTE 2: CENÁRIO B — Demonstração de data leakage
    # ------------------------------------------------------------------
    print("\n--- DEMONSTRAÇÃO DE DATA LEAKAGE (Cenário B) ---")
    X_train_B, _, y_train_B, _, feature_names_B = get_prepared_data(cenario='B')

    models_B = get_models()
    trained_models_B = {}
    for name, pipe in models_B.items():
        pipe.fit(X_train_B, y_train_B)
        trained_models_B[name] = pipe

    bundle_B = {
        "modelos":       trained_models_B,
        "feature_order": feature_names_B,
        "degree_cols":   [c for c in feature_names_B if c.startswith("Degree_")],
        "tem_leak":      True,
    }
    joblib.dump(bundle_B, MODEL_DIR / "bundle_com_leak.pkl")
    print("=> bundle_com_leak.pkl salvo em model_artifacts/")

    print("\n Pipeline Finalizado com Sucesso!")

if __name__ == '__main__':
    run_pipeline()
