import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

file_path = './data/raw/Student Depression Dataset.csv'

def get_prepared_data(cenario='A'):

    df = pd.read_csv(file_path)

    # Filtragem: apenas estudantes + remoção de colunas inúteis
    df = df[df['Profession'] == 'Student'].copy()
    df.drop(columns=['Profession', 'City', 'Work Pressure', 'Job Satisfaction', 'id'],
            inplace=True, errors='ignore')

    # Encoding de variáveis binárias
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    df['Have you ever had suicidal thoughts ?'] = df['Have you ever had suicidal thoughts ?'].map({'No': 0, 'Yes': 1})
    df['Family History of Mental Illness'] = df['Family History of Mental Illness'].map({'No': 0, 'Yes': 1})

    # Sleep Duration (ordinal) — 'Others' vira NaN
    sleep_map = {
        'Less than 5 hours': 0,
        '5-6 hours': 1,
        '7-8 hours': 2,
        'More than 8 hours': 3,
        'Others': np.nan,
    }
    df['Sleep Duration'] = df['Sleep Duration'].map(sleep_map)

    # Dietary Habits (ordinal) — 'Others' vira NaN
    diet_map = {
        'Unhealthy': 0,
        'Moderate': 1,
        'Healthy': 2,
        'Others': np.nan,
    }
    df['Dietary Habits'] = df['Dietary Habits'].map(diet_map)

    # One-Hot Encoding de Degree
    df = pd.get_dummies(df, columns=['Degree'], drop_first=True)
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    # Feature Engineering
    df['High_Academic_Pressure'] = (df['Academic Pressure'] >= 4).astype(int)
    df['High_Financial_Stress'] = (df['Financial Stress'] >= 4).astype(int)

    # Definição dos cenários (com/sem data leakage)
    target = 'Depression'
    leak_col = 'Have you ever had suicidal thoughts ?'

    X_full = df.drop(columns=[target]) 
    y = df[target]

    if cenario == 'A':
        X = X_full.drop(columns=[leak_col])  
    else:
        X = X_full                            

    # Split 80/20 estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    feature_names = list(X.columns)

    print(f"Shape X: {X.shape} | Treino: {X_train.shape} | Teste: {X_test.shape}")
    print(f"Proporção classe positiva — treino: {y_train.mean():.3f} | teste: {y_test.mean():.3f}")

    return X_train, X_test, y_train, y_test, feature_names
