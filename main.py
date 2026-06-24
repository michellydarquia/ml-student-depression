from src.preprocessing import get_prepared_data
#from src.models.knn import train_knn
#from src.models.regressao import train_regressao
#from src.models.decision_tree import train_decision_tree


# Cenario B = Usamos a coluna  "Have you ever had suicidal thoughts ?"
# Cenario A = Não usamos essa coluna, pois os algoritmos podem aprender so olhando para ela (Data Leakage)


def run_pipeline():
    print("Iniciando o Pipeline de Machine Learning...")

    # ---------------------------------------------------------
    # PARTE 1: O PROJETO OFICIAL (CENÁRIO A - Sem Vazamento)
    # ---------------------------------------------------------

    print("[1/4] Carregando e pré-processando dados (Cenário A)...")
    x_train_a, x_test_a, y_train_a, y_test_a = get_prepared_data(cenario='A')




    print("\n[2/4] Treinando Modelos...")

    #     knn_model = train_knn(x_train_a, y_train_a)
    #     svm_model = train_regressao(x_train_a, y_train_a)
    #     tree_model = train_decision_tree(x_train_a, y_train_a)
    #
    #

    print("\n[3/4] Avaliando Modelos e gerando Matrizes...")
    
    # 
    
    print("\n[4/4] Rodando Teste de Friedman e Wilcoxon...")

    #

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