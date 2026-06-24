# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="Predição de Depressão em Estudantes", layout="centered")

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "model_artifacts"

print(MODEL_DIR)

@st.cache_resource
def carregar_bundles():
    return {
        "Sem a variável-alvo de risco (cenário principal)": joblib.load(MODEL_DIR / "bundle_sem_leak.pkl"),
        "Com a variável de pensamentos suicidas (cenário comparativo)": joblib.load(MODEL_DIR / "bundle_com_leak.pkl"),
    }

bundles = carregar_bundles()

# mapeamentos -> igual ao do notebook
SLEEP_MAP = {
    "Menos de 5 horas": 0,
    "5-6 horas": 1,
    "7-8 horas": 2,
    "Mais de 8 horas": 3,
    "Outros": np.nan,
}
DIET_MAP = {
    "Não saudável": 0,
    "Moderada": 1,
    "Saudável": 2,
    "Outros": np.nan,
}

st.title("Predição de Depressão em Estudantes")
st.caption("Projeto de Machine Learning: classificação binária (0 = sem depressão, 1 = com depressão)")

# escolher cenário e modelo
st.sidebar.header("Configuração")

nome_cenario = st.sidebar.selectbox("Cenário", list(bundles.keys()))
bundle = bundles[nome_cenario]

nome_modelo = st.sidebar.selectbox("Modelo", list(bundle["modelos"].keys()))
modelo = bundle["modelos"][nome_modelo]

feature_order = bundle["feature_order"]
degree_cols = bundle["degree_cols"]
tem_leak = bundle["tem_leak"]

# lista de cursos (Degree) disponíveis = dummies + a categoria base (drop_first)
cursos = [c.replace("Degree_", "") for c in degree_cols]
cursos_opcoes = ["(curso base / outro)"] + sorted(cursos)

st.sidebar.markdown("---")
st.sidebar.write(f"**Features esperadas:** {len(feature_order)}")
if tem_leak:
    st.sidebar.warning("Este cenário inclui a variável de pensamentos suicidas, "
                       "que pode causar *vazamento de dados* (target leakage). "
                       "Use apenas para fins comparativos.")

# Formulário de entrada
st.subheader("Responda às perguntas")

with st.form("formulario"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Idade", min_value=18, max_value=59, value=25, step=1)
        gender = st.selectbox("Gênero", ["Masculino", "Feminino"])
        academic_pressure = st.slider("Pressão acadêmica (1-5)", 1, 5, 3)
        study_satisfaction = st.slider("Satisfação com os estudos (1-5)", 1, 5, 3)
        cgpa = st.number_input("CGPA (0-10)", min_value=0.0, max_value=10.0, value=7.5, step=0.1)

    with col2:
        financial_stress = st.slider("Estresse financeiro (1-5)", 1, 5, 3)
        work_study_hours = st.number_input("Horas de estudo/trabalho por dia", min_value=0, max_value=24, value=6, step=1)
        sleep = st.selectbox("Duração do sono", list(SLEEP_MAP.keys()))
        diet = st.selectbox("Hábitos alimentares", list(DIET_MAP.keys()))
        family_history = st.selectbox("Histórico familiar de doença mental", ["Não", "Sim"])

    degree = st.selectbox("Curso (Degree)", cursos_opcoes)

    # variável de vazamento só aparece no cenário com leak
    if tem_leak:
        suicidal = st.selectbox("Já teve pensamentos suicidas?", ["Não", "Sim"])

    enviar = st.form_submit_button("Prever")

# montagem do vetor de features na ORDEM correta
def montar_entrada():
    dados = {col: 0 for col in feature_order}  # inicializa tudo com 0

    # numéricas / ordinais
    dados["Age"] = age
    dados["Gender"] = 0 if gender == "Masculino" else 1
    dados["Academic Pressure"] = academic_pressure
    dados["CGPA"] = cgpa
    dados["Study Satisfaction"] = study_satisfaction
    dados["Sleep Duration"] = SLEEP_MAP[sleep]
    dados["Dietary Habits"] = DIET_MAP[diet]
    dados["Work/Study Hours"] = work_study_hours
    dados["Financial Stress"] = financial_stress
    dados["Family History of Mental Illness"] = 0 if family_history == "Não" else 1

    # engenharia de atributos -> igual ao do notebook
    dados["High_Academic_Pressure"] = int(academic_pressure >= 4)
    dados["High_Financial_Stress"] = int(financial_stress >= 4)

    # one-hot do Degree
    if degree != "(curso base / outro)":
        col = f"Degree_{degree}"
        if col in dados:
            dados[col] = 1

    # variável de vazamento (apenas cenário com leak)
    if tem_leak:
        dados["Have you ever had suicidal thoughts ?"] = 0 if suicidal == "Não" else 1

    # garante a ordem exata
    X_in = pd.DataFrame([[dados[c] for c in feature_order]], columns=feature_order)
    return X_in

# predição

if enviar:
    X_in = montar_entrada()
    pred = modelo.predict(X_in)[0]

    st.markdown("---")
    st.subheader("Resultado")

    if pred == 1:
        st.error("**Indicativo: COM depressão (classe 1)**")
    else:
        st.success("**Indicativo: SEM depressão (classe 0)**")

    # probabilidade, se o modelo suportar
    if hasattr(modelo, "predict_proba"):
        proba = modelo.predict_proba(X_in)[0][1]
        st.metric("Probabilidade estimada de depressão", f"{proba*100:.1f}%")
        st.progress(float(proba))

    st.caption(f"Modelo: **{nome_modelo}** | Cenário: **{nome_cenario}**")

    with st.expander("Ver vetor de features enviado ao modelo"):
        st.dataframe(X_in.T.rename(columns={0: "valor"}))

    st.info(" Ferramenta acadêmica/demonstrativa. Não substitui avaliação profissional de saúde mental.")
