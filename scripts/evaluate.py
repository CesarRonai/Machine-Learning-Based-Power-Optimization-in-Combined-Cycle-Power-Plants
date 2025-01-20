import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from preprocess import load_and_preprocess_data

def evaluate_model():
    """Load model and evaluate it."""
    df = load_and_preprocess_data("C:/Users/Usuário/CCPP/Folds5x2_pp.xlsx")

    X = df.drop(columns=["PE"])
    y = df["PE"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Carregar o modelo treinado
    model = joblib.load("C:/Users/Usuário/Machine-Learning-Based-Power-Optimization-in-Combined-Cycle-Power-Plants\models\modelo_xgboost.pkl")

    # Fazer previsões
    y_pred = model.predict(X_test)

    # Avaliação
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MSE: {mse:.2f}")
    print(f"R²: {r2:.2f}")

if __name__ == "__main__":
    evaluate_model()
