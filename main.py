import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import f1_score

# Importando os seus módulos
from src.preprocessing import get_prepared_data
from src.statistical_tests import run_statistical_tests
from src.visualizations import plot_feature_importance, plot_confusion_matrices, plot_roc_curves, plot_cv_boxplot,report_best_model,  plot_logistic_regression_coefs
from src.models.models import get_models  

# Cenario B = Usamos a coluna  "Have you ever had suicidal thoughts ?"
# Cenario A = Não usamos essa coluna, pois os algoritmos podem aprender so olhando para ela (Data Leakage)

def run_pipeline():
    print("Iniciando o Pipeline de Machine Learning...")

    # ---------------------------------------------------------
    # PARTE 1: O PROJETO OFICIAL (CENÁRIO A - Sem Vazamento)
    # ---------------------------------------------------------

    print("[1/4] Carregando e pré-processando dados (Cenário A)...")
    # Nota: 'cenario' foi mantido em português pois é o nome do parâmetro lá no preprocessing.py
    X_train_A, X_test_A, y_train_A, y_test_A, feature_names_A = get_prepared_data(cenario='A')

    print("\n[2/4] Treinando Modelos...")
    models = get_models()

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    
    cv_raw = {}
    trained_models = {}
    cv_summary_rows = []

    print("\nResultados da Validação Cruzada (10 Folds):")
    
    for name, model in models.items():
        # Validação cruzada
        res = cross_validate(model, X_train_A, y_train_A, cv=cv, scoring=scoring, n_jobs=-1)
        cv_raw[name] = res
        
        row = {"Modelo": name}
        for s in scoring:
            row[f"{s}_mean"] = res[f"test_{s}"].mean()
            row[f"{s}_std"] = res[f"test_{s}"].std()
        cv_summary_rows.append(row)
        
        print(f"{name:22s} | acc={row['accuracy_mean']:.4f}  f1={row['f1_mean']:.4f}  auc={row['roc_auc_mean']:.4f}")

        # Treino no conjunto inteiro
        model.fit(X_train_A, y_train_A)
        trained_models[name] = model

    # 4. Cria e exibe o DataFrame ordenado 
    cv_summary = pd.DataFrame(cv_summary_rows).sort_values("f1_mean", ascending=False).reset_index(drop=True)
    print("\nResumo da Validação Cruzada (Ordenado por F1-Score):")
    print(cv_summary[["Modelo", "f1_mean", "accuracy_mean", "roc_auc_mean"]])

    print("\n[3/4] Avaliando Modelos e gerando Matrizes...")
    
    plot_confusion_matrices(trained_models, X_test_A, y_test_A)
    plot_roc_curves(trained_models, X_test_A, y_test_A)
    plot_cv_boxplot(cv_raw)
    plot_logistic_regression_coefs(trained_models["Regressão Logística"], feature_names_A)
    
    if "Random Forest" in trained_models:
        plot_feature_importance(trained_models["Random Forest"], feature_names_A, "rf_importances_cenarioA.png")
    

    print("\n[4/4] Rodando Teste de Friedman e Wilcoxon...")

    df_statistics = run_statistical_tests(cv_raw, metric_for_test="test_f1")

    df_statistics.to_csv("resultados/tabela_wilcoxon_bonferroni.csv", index=False)
    print("=> Resultados estatísticos salvos em 'tabela_wilcoxon_bonferroni.csv'")

    # Salvando o modelo do Cenário A
    bundle_A = {
        "modelos": trained_models, # A chave do dicionário continuou 'modelos' para não quebrar o seu carregamento futuro
        "feature_order": feature_names_A,
        "degree_cols": [c for c in feature_names_A if c.startswith("Degree_")],
        "tem_leak": False,
    }
    joblib.dump(bundle_A, "bundle_sem_leak.pkl")
    print("\n=> Pipeline da Parte 1 Finalizado com Sucesso! (Modelos salvos em .pkl)")
    

    print("\n Pipeline Finalizado com Sucesso!")


    # ---------------------------------------------------------
    # PARTE 2: A PROVA DO CRIME (Comparação com Cenário B)
    # ---------------------------------------------------------
    print("\n--- DEMONSTRAÇÃO DE DATA LEAKAGE (Cenário B) ---")
    X_train_B, X_test_B, y_train_B, y_test_B, feature_names_B = get_prepared_data(cenario='B')
    
    # Aqui nós NÃO rodamos os 6 modelos de novo. Escolhemos apenas 1 (ex: Random Forest)
    # Treinamos ele com os dados do Cenário B e mostramos pro professor:
    # "Olha professor, no cenário A o modelo acertou 85%. No Cenário B ele acertou 99%."
    # E aí entra o script final de comparação só para exibir essa diferença na tela!


if __name__ == '__main__':
    run_pipeline()