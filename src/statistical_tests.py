# statistical_tests.py

import numpy as np
import pandas as pd
import scipy.stats as stats

def run_statistical_tests(cv_raw, metric_for_test="test_f1"):
    """
    Testes de Friedman e Wilcoxon para comparar os modelos
    usando os resultados (cv_raw) da validação cruzada.
    """
    print(f"Métrica: {metric_for_test})")

    model_names = list(cv_raw.keys())

    # matriz (n_modelos x n_dobras)
    fold_matrix = np.array([cv_raw[name][metric_for_test] for name in model_names])

    # Teste de Friedman
    friedman_stat, friedman_p = stats.friedmanchisquare(*fold_matrix)
    print(f"Teste de Friedman (10 dobras):")
    print(f"  estatística = {friedman_stat:.4f}")
    print(f"  p-valor     = {friedman_p:.4g}")

    if friedman_p < 0.05:
        print("=> Existe diferença estatisticamente significativa entre pelo menos dois modelos.")
    else:
        print("=> Não há evidência de diferença significativa entre os modelos.")

    # Comparações pareadas (Wilcoxon) com correção de Bonferroni
    n_models = len(model_names)
    n_comparisons = n_models * (n_models - 1) // 2
    alpha = 0.05
    alpha_bonferroni = alpha / n_comparisons

    paired_results = []
    for i in range(n_models):
        for j in range(i + 1, n_models):
            try:
                stat_w, p_w = stats.wilcoxon(fold_matrix[i], fold_matrix[j])
            except ValueError:
                stat_w, p_w = np.nan, 1.0
                
            paired_results.append({
                "Modelo A": model_names[i],
                "Modelo B": model_names[j],
                "estatistica": stat_w,
                "p_valor": p_w,
                "significativo (Bonferroni)": p_w < alpha_bonferroni
            })

    df_paired = pd.DataFrame(paired_results).sort_values("p_valor").reset_index(drop=True)
    print(f"\nalpha original = {alpha} | alpha corrigido (Bonferroni, {n_comparisons} comparações) = {alpha_bonferroni:.5f}\n")
    print(df_paired)
    
    return df_paired