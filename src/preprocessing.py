import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


file_path = './data/raw/Student Depression Dataset.csv'
    

def get_prepared_data(cenario='A'):

    df = pd.read_csv(file_path)

    # 1 -- O Problema do Ruído no Público-Alvo e O Problema das Variáveis Inúteis e Quebradas

    # Pegando apenas os estudantes --- depois vira ruid opois todos serao estudantes
    df = df[df['Profession'] == 'Student']
    
    df.drop(columns=['Profession', 'City', 'Work Pressure', 'Job Satisfaction', 'id'], inplace=True) # intrace truque para excluir no objeto direto do pandas


    # 2 -- O Problema dos Dados Ausentes

    mediana_stress = df['Financial Stress'].median()

    df['Financial Stress'] = df['Financial Stress'].fillna(mediana_stress)

    # 3 -- Feature Engineering)

    ## Tecnica de Encoding 
    df['Gender'] = df['Gender'].map({'Male':0 , 'Female': 1})
    df['Have you ever had suicidal thoughts ?'] = df['Have you ever had suicidal thoughts ?'].map({'No':0 , 'Yes': 1})
    df['Sleep Duration'] = df['Sleep Duration'].map({'Less than 5 hours': 0, '5-6 hours': 1, '7-8 hours': 2, 'More than 8 hours': 3})
    df['Dietary Habits'] = df ['Dietary Habits'].map({'Unhealthy': 0, 'Moderate': 1, 'Healthy': 2})
    df['Family History of Mental Illness'] = df['Family History of Mental Illness'].map({'No': 0, 'Yes': 1})

    ## Tecnica de One-Hot Encoding 

    df = pd.get_dummies(df, columns=['Degree'], drop_first=True, dtype=int) ## drop apaga a coluna Degree e dtype preenche com 0 ou 1
    
    
    if cenario =='A': # Cenario A = Não usamos essa coluna, pois os algoritmos podem aprender so olhando para ela (Data Leakage)
        df.drop(columns = ['Have you ever had suicidal thoughts ?'], inplace = True) # Cenario B = usamos a coluna
    
    
    y = df['Depression']
    X = df.drop(columns=['Depression'])
    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    nomes_das_features = list(X.columns)

    imputer = SimpleImputer(strategy='median')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    ## Mudando ESCALAS USANDO SCALER
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test) # O escalonador NÃO PODE olhar para os dados de teste. Se ele olhar, ele "rouba" a média do futuro.


    print(df.head())
    return  X_train, X_test, y_train, y_test, nomes_das_features
    
# get_prepared_data('A')