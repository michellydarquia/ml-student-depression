from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


def build_pipeline(model, scale):
    """Monta um Pipeline com imputação (sempre) + escala opcional + classificador."""
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def get_models():
    models = {
        "Regressão Logística": build_pipeline(
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE), scale=True),
        "KNN": build_pipeline(
            KNeighborsClassifier(n_jobs=-1), scale=True),
        "Árvore de Decisão": build_pipeline(
            DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE), scale=False),
        "Random Forest": build_pipeline(
            RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1), scale=False),
        "Naive Bayes": build_pipeline(
            GaussianNB(), scale=False),
        # "SVM": build_pipeline(
        #     SVC(probability=True, random_state=RANDOM_STATE), scale=True),
    }
    return models
