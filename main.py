import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import f1_score

# Importando os seus módulos
from src.preprocessing import get_prepared_data
from src.statistical_tests import run_statistical_tests
# from src.visualizations import plot_feature_importance, plot_confusion_matrices, plot_roc_curves
from src.models.models import get_models  

# Cenario B = Usamos a coluna  "Have you ever had suicidal thoughts ?"
# Cenario A = Não usamos essa coluna, pois os algoritmos podem aprender so olhando para ela (Data Leakage)


def run_pipeline():
    print("Iniciando o Pipeline de Machine Learning...")

    # ---------------------------------------------------------
    # PARTE 1: O PROJETO OFICIAL (CENÁRIO A - Sem Vazamento)
    # ---------------------------------------------------------

    print("[1/4] Carregando e pré-processando dados (Cenário A)...")
    X_train_A, X_test_A, y_train_A, y_test_A, feature_names_A = get_prepared_data(cenario='A')

    print("\n[2/4] Treinando Modelos...")
    modelos = get_models()

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    
    cv_raw = {}
    treinados = {}
    cv_summary_rows = []

    print("\nResultados da Validação Cruzada (10 Folds):")
    
    for name, model in modelos.items():
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
        treinados[name] = model

    # 4. Cria e exibe o DataFrame ordenado 
    cv_summary = pd.DataFrame(cv_summary_rows).sort_values("f1_mean", ascending=False).reset_index(drop=True)
    print("\nResumo da Validação Cruzada (Ordenado por F1-Score):")
    print(cv_summary[["Modelo", "f1_mean", "accuracy_mean", "roc_auc_mean"]])

    print("\n[3/4] Avaliando Modelos e gerando Matrizes...")
    

    print("\n Pipeline Finalizado com Sucesso!")


    # ---------------------------------------------------------
    # PARTE 2: A PROVA DO CRIME (Comparação com Cenário B)
    # ---------------------------------------------------------
    print("\n--- DEMONSTRAÇÃO DE DATA LEAKAGE (Cenário B) ---")
    X_train_B, X_test_B, y_train_B, y_test_B = get_prepared_data(cenario='B')
    
    # Aqui nós NÃO rodamos os 6 modelos de novo. Escolhemos apenas 1 (ex: Random Forest)
    # Treinamos ele com os dados do Cenário B e mostramos pro professor:
    # "Olha professor, no cenário A o modelo acertou 85%. No Cenário B ele acertou 99%."
    # E aí entra o script final de comparação só para exibir essa diferença na tela!


if __name__ == '__main__':
    run_pipeline()