import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from preprocess import load_and_preprocess_data

def train_and_save_model():
    """Train the model and save it."""
    df = load_and_preprocess_data("C:/Users/Usuário/CCPP/Folds5x2_pp.xlsx")



    # Separar features e target
    X = df.drop(columns=["PE"])
    y = df["PE"]

    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Treinar modelo
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Salvar o modelo treinado
    joblib.dump(model, "C:/Users/Usuário/Machine-Learning-Based-Power-Optimization-in-Combined-Cycle-Power-Plants\models\modelo_xgboost.pkl")
    print("Modelo salvo em '../models/best_model.pkl'.")

if __name__ == "__main__":
    train_and_save_model()
