# visualizations.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, classification_report

OUTPUT_DIR = "resultados"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_confusion_matrices(trained_models, X_test, y_test):
    """Gera o painel com as matrizes de confusão de todos os modelos."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for ax, name in zip(axes, trained_models.keys()):
        ConfusionMatrixDisplay.from_estimator(
            trained_models[name], X_test, y_test, ax=ax,
            display_labels=["Sem depressão", "Com depressão"],
            colorbar=False,
            cmap='Blues'
        )
        ax.set_title(name)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "matrizes_de_confusao.png"), dpi=300)
    plt.show()

def plot_roc_curves(trained_models, X_test, y_test):
    """Gera o gráfico comparativo das Curvas ROC."""
    fig, ax = plt.subplots(figsize=(8, 7))

    for name, model in trained_models.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=name)

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Classificador aleatório")
    ax.set_title("Curvas ROC - Comparação entre os algoritmos")
    ax.legend(loc="lower right", fontsize=9)
    plt.savefig(os.path.join(OUTPUT_DIR, "curvas_roc.png"), dpi=300)
    plt.show()

def report_best_model(best_model, model_name, X_test, y_test):
    print(f"\n--- Relatório do Melhor Modelo: {model_name} ---")
    y_pred = best_model.predict(X_test)
    print(classification_report(
        y_test, y_pred,
        target_names=["Sem depressão", "Com depressão"]
    ))

def plot_logistic_regression_coefs(logreg_model, feature_names):
    """Plota os coeficientes da Regressão Logística para interpretabilidade."""
    coefficients = pd.Series(logreg_model.coef_[0], index=feature_names)
    sorted_coefficients = coefficients.reindex(coefficients.abs().sort_values(ascending=False).index)

    plt.figure(figsize=(10, 6))
    top_coefs = sorted_coefficients.head(15).sort_values()
    
    colors = ["indianred" if v > 0 else "seagreen" for v in top_coefs]
    top_coefs.plot(kind="barh", color=colors)
    
    plt.title("Top 15 coeficientes - Regressão Logística\n(Vermelho = aumenta risco | Verde = reduz risco)")
    plt.xlabel("Coeficiente (dados padronizados)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "coeficientes_regressao_logistica.png"), dpi=300)
    plt.show()

def plot_feature_importance(rf_model, feature_names, output_filename="feature_importance.png"):
    """Extrai e plota a importância das variáveis do modelo Random Forest."""
    if not hasattr(rf_model, "feature_importances_"):
        print("Erro: O modelo não possui o atributo 'feature_importances_'.")
        return

    importances = pd.Series(rf_model.feature_importances_, index=feature_names).sort_values(ascending=False)

    plt.figure(figsize=(8, 6))
    importances.head(15).sort_values().plot(kind="barh", color="steelblue")
    plt.title("Top 15 variáveis mais importantes - Random Forest")
    plt.xlabel("Importância (Gini)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, output_filename), dpi=300)
    plt.show()
    print(f"\nGráfico salvo como '{output_filename}'")

def plot_cv_boxplot(cv_raw):
    
    # Extrai os nomes dos modelos e os arrays de test_f1
    f1_per_model = pd.DataFrame({name: cv_raw[name]["test_f1"] for name in cv_raw})

    plt.figure(figsize=(10, 5))
    sns.boxplot(data=f1_per_model, orient="h", palette="Set2")
    plt.title("Distribuição do F1-score nas 10 dobras da validação cruzada")
    plt.xlabel("F1-score")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "boxplot_f1_cv.png"), dpi=300)
    plt.show()