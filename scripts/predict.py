import joblib
import pandas as pd

def make_prediction(new_data):
    """Predict power output given new input data."""
    model = joblib.load("C:/Users/Usuário/CCPP/Folds5x2_pp.xlsx")
    prediction = model.predict(new_data)
    return prediction

if __name__ == "__main__":
    # Exemplo de novo dado (substituir pelos valores reais)
    new_data = pd.DataFrame([[14.96, 41.76, 1024.07, 73.17]], 
                            columns=["AT", "V", "AP", "RH"])
    
    prediction = make_prediction(new_data)
    print(f"Predicted Power Output: {prediction[0]:.2f} MW")
